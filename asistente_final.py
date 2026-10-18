import streamlit as st
import os
import re
import json
from datetime import datetime
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.memory import ConversationBufferWindowMemory

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Agente de Liquidación - ReAct", page_icon="⚖️", layout="wide")
st.title("⚖️ Agente Autónomo de Liquidación de Siniestros")
st.caption("Arquitectura ReAct Multietapa | Embeddings: nomic-embed-text | LLM: Hermes 3")

# --- 2. UTILIDADES ---
_RE_ID = re.compile(r"CASO\s*[-_]?\s*(\d{1,4})(?!\d)", re.IGNORECASE)


def _detectar_id(texto: str):
    """Detecta 'CASO-002', 'caso 002', 'caso 2' o 'caso002' y lo normaliza a 'CASO-002'. None si no hay."""
    m = _RE_ID.search(str(texto))
    return f"CASO-{int(m.group(1)):03d}" if m else None


def _extraer_id(texto: str) -> str:
    """Normaliza el ID aunque el modelo envíe texto adicional, comillas o un formato distinto."""
    return _detectar_id(texto) or str(texto).strip().upper()


def _leer_json(ruta: str, defecto):
    if not os.path.exists(ruta):
        return defecto
    try:
        with open(ruta, "r", encoding="utf-8-sig") as f:  # utf-8-sig tolera BOM
            return json.load(f)
    except Exception:
        return defecto


def _md(texto) -> str:
    """Escapa '$' para que Streamlit no lo interprete como fórmula LaTeX (rompía los montos en pantalla)."""
    return str(texto).replace("$", "\\$")


# --- 3. VECTORSTORE PARA RAG SEMÁNTICO (IE4) ---
@st.cache_resource(show_spinner="Indexando póliza con embeddings...")
def inicializar_vectorstore():
    if not os.path.exists("poliza.txt"):
        return None
    try:
        with open("poliza.txt", "r", encoding="utf-8") as f:
            texto = f.read()

        patron = re.compile(
            r"^ARTÍCULO\s+(\d+)\s*:\s*(.+?)(?=^ARTÍCULO\s+\d+\s*:|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        documentos = [
            Document(page_content=f"ARTÍCULO {n}: {c.strip()}", metadata={"articulo": int(n)})
            for n, c in patron.findall(texto)
        ]
        if not documentos:
            documentos = [Document(page_content=l.strip()) for l in texto.splitlines() if l.strip()]

        return Chroma.from_documents(
            documents=documentos,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="normativa_poliza",
        )
    except Exception as e:
        st.error(f"No se pudo crear el índice vectorial: {e}")
        return None


vectorstore_poliza = inicializar_vectorstore()

# --- 3b. ANÁLISIS DEL RELATO E IDENTIFICACIÓN SEMÁNTICA (IE4, IE5) ---
_RE_SINIESTRO = re.compile(
    r"rob(o|a|e|ar|aron|ado|ada|ó)\b|hurt|sustra|choc|colisi|incendi|quem(o|ó|aron)|"
    r"siniestro|accidente|volc|inund|\bauto\b|autom[oó]vil|veh[ií]culo|camioneta|"
    r"\bradio\b|espejo|estacionamiento|reclam|denunci",
    re.IGNORECASE,
)
_RE_REFINA = re.compile(r"completo|entero|accesorio|llevaron|\bsolo\b|\bsólo\b|únicamente|nada m[aá]s", re.IGNORECASE)
_RE_CONFIRMA = re.compile(
    r"^\s*(s[ií]|correcto|exacto|confirmo|ese|as[ií] es|ok|dale|claro)\b", re.IGNORECASE
)
_RE_NEGACION = re.compile(r"\bno\b", re.IGNORECASE)

_VEH = r"(auto|autom[oó]vil|veh[ií]culo|camioneta|coche|carro)"
_RE_INCENDIO = re.compile(r"incendi|quem[oó]|quemaron|se prendi|fuego|llamas", re.IGNORECASE)
_RE_COLISION = re.compile(r"chocaron|choque|choqu[eé]|choc[oó]|colisi|accidente|impact|embistieron", re.IGNORECASE)
_RE_ROBO = re.compile(r"\brob(o|ó|aron|ado|ada|ar|an)\b|hurt|sustra|se llevaron|se lo llevaron|asalt", re.IGNORECASE)
_RE_ROBO_TOTAL = re.compile(
    r"robo total|sustracci[oó]n completa|se lo llevaron todo|"
    r"(robaron|robo|robado|llevaron|sustrajeron|roban)\s+(de\s+)?(mi|el|un|todo el|todo mi)?\s*" + _VEH + r"\b|"
    + _VEH + r"\s+(completo|entero)|"
    + _VEH + r"\s+(me\s+)?(lo\s+|fue\s+|ha sido\s+)?(robad[oa]|robaron|sustra[ií]d[oa])",
    re.IGNORECASE,
)
_RE_ACC_RELATO = re.compile(
    r"\b(radio|est[eé]reo|panel|espejos?|accesorios?|parlantes?|llantas?|ruedas?|neum[aá]ticos?|pantalla|bocina)\b",
    re.IGNORECASE,
)

_ETIQUETAS = {
    "robo total": "un robo total del vehículo",
    "robo parcial": "un robo de accesorios (robo parcial)",
    "colisión": "una colisión",
    "incendio": "un incendio",
    "robo": "un robo (aún sin precisar si fue del vehículo completo o de accesorios)",
}


def _parece_siniestro(texto: str) -> bool:
    return bool(_RE_SINIESTRO.search(str(texto)))


def _es_confirmacion(texto: str) -> bool:
    return bool(_RE_CONFIRMA.search(str(texto))) and not _RE_NEGACION.search(str(texto))


def _clasificar(texto: str):
    t = str(texto)
    if _RE_INCENDIO.search(t):
        return "incendio"
    if _RE_COLISION.search(t):
        return "colisión"
    if _RE_ROBO_TOTAL.search(t):
        return "robo total"
    if _RE_ACC_RELATO.search(t):
        return "robo parcial"
    if _RE_ROBO.search(t):
        return "robo"
    return None


def _compatible(tipo_relato, tipo_expediente) -> bool:
    if tipo_relato == tipo_expediente:
        return True
    return tipo_relato == "robo" and tipo_expediente in ("robo total", "robo parcial")


def _tipo_del_relato(texto: str):
    tipo = _clasificar(texto)
    if tipo or not _parece_siniestro(texto):
        return tipo
    try:
        r = llm.invoke(PROMPT_CLASIFICAR.format(relato=texto)).content.lower()
    except Exception:
        return None
    for etiqueta in ("robo total", "robo parcial", "colisión", "colision", "incendio"):
        if etiqueta in r:
            return "colisión" if etiqueta == "colision" else etiqueta
    return None


def _obtener_caso(id_caso: str):
    try:
        return next((c for c in _leer_json("casos.json", []) if c.get("id") == id_caso), None)
    except Exception:
        return None


def _clase_caso(caso) -> str:
    return _clasificar(f"{caso['tipo']} {caso['descripcion']}") if caso else None


@st.cache_resource(show_spinner="Indexando expedientes con embeddings...")
def _cargar_indice_casos():
    casos = _leer_json("casos.json", [])
    documentos = [
        Document(page_content=f"{c['tipo']}. {c['descripcion']}", metadata={"id": c["id"]})
        for c in casos
    ]
    return Chroma.from_documents(
        documents=documentos,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="expedientes",
    )


def _ranking_lexico(relato: str, casos: list):
    def tokens(t):
        return {w for w in re.findall(r"[a-záéíóúñ]{4,}", str(t).lower())}
    base = tokens(relato)
    puntajes = []
    for c in casos:
        solape = len(base & tokens(f"{c['tipo']} {c['descripcion']}"))
        if solape:
            puntajes.append((c["id"], -float(solape)))
    return sorted(puntajes, key=lambda x: x[1])


def identificar_caso_por_relato(relato: str):
    try:
        casos = _leer_json("casos.json", [])
    except Exception:
        return []
    if not casos:
        return []
    try:
        resultados = _cargar_indice_casos().similarity_search_with_score(relato, k=len(casos))
        orden = [(d.metadata["id"], float(s)) for d, s in resultados]
    except Exception:
        orden = _ranking_lexico(relato, casos)
    por_id = {c["id"]: c for c in casos}
    vistos = {i for i, _ in orden}
    fichas = [(por_id[i], s) for i, s in orden if i in por_id]
    fichas += [(c, None) for c in casos if c["id"] not in vistos]
    return [
        {"id": c["id"], "tipo": c["tipo"], "descripcion": c["descripcion"],
         "clase": _clase_caso(c), "puntaje": s}
        for c, s in fichas
    ]


def _ranking_por_tipo(relato: str, tipo):
    todos = identificar_caso_por_relato(relato)
    if tipo:
        compat = [c for c in todos if _compatible(tipo, c["clase"])]
        descartados = [c for c in todos if not _compatible(tipo, c["clase"])]
    else:
        compat = [c for c in todos if c["puntaje"] is not None]
        descartados = []
    return compat[:3], descartados


def _pregunta_aclaratoria(tipo):
    if tipo == "robo":
        return "¿Se llevaron el vehículo completo o solo accesorios (radio, espejos, etc.)?"
    if tipo is None:
        return "¿Qué ocurrió exactamente: un robo, una colisión o un incendio?"
    return "¿Cuál se parece más a lo que te ocurrió? Responde 1, 2 o cuéntame más detalles."


def _mensaje_pregunta(tipo, compat, pregunta) -> str:
    if tipo:
        partes = [f"Entiendo que se trata de **{_ETIQUETAS[tipo]}**."]
    else:
        partes = ["Quiero identificar tu caso, pero necesito un dato más."]
    if tipo not in (None, "robo") and len(compat) > 1:
        partes += ["", "Hay más de un expediente de ese tipo:"]
        for n, c in enumerate(compat, 1):
            partes.append(f"{n}. **{c['id']}** — {c['tipo']}: {c['descripcion']}")
    partes += ["", pregunta]
    return "\n".join(partes)


def _elegir_opcion(texto: str, opciones):
    if not opciones or len(str(texto).strip()) > 25:
        return None
    m = re.search(r"\b([1-9])\b", texto)
    if m:
        n = int(m.group(1))
    else:
        n = next((v for k, v in {"primer": 1, "segund": 2, "tercer": 3}.items()
                  if re.search(k, texto, re.IGNORECASE)), None)
    return opciones[n - 1] if n and 1 <= n <= len(opciones) else None


def _hay_discrepancia(tipo_relato, caso) -> bool:
    tipo_exp = _clase_caso(caso)
    return bool(tipo_relato and tipo_exp and not _compatible(tipo_relato, tipo_exp))


def _mensaje_discrepancia(tipo_relato, caso, mejor) -> str:
    texto = (f"Ojo: tu relato describe **{_ETIQUETAS[tipo_relato]}**, pero el expediente **{caso['id']}** "
             f"corresponde a **{_ETIQUETAS[_clase_caso(caso)]}** ({caso['descripcion']})")
    if mejor:
        texto += f"\n\nSegún tu relato, el más probable sería **{mejor['id']}**: {mejor['descripcion']}"
    texto += (f"\n\nResponde **sí** para procesar {caso['id']} de todos modos, "
              "o cuéntame más / indícame otro ID.")
    return texto


def _nota_discrepancia(tipo_relato, caso) -> str:
    return (f"⚠️ Nota: tu relato describe {_ETIQUETAS[tipo_relato]}, pero el expediente {caso['id']} "
            f"corresponde a {_ETIQUETAS[_clase_caso(caso)]}. El dictamen se basa en los datos del expediente.")


# --- 4. HERRAMIENTAS (IE1, IE3, IE4) ---
@tool
def obtener_expediente_caso(id_caso: str) -> str:
    """Consulta la base de datos de siniestros 'casos.json' usando el ID (ej: CASO-001). Devuelve los datos del siniestro."""
    id_caso = _extraer_id(id_caso)
    try:
        casos = _leer_json("casos.json", None)
        if casos is None:
            return "Error: No existe el repositorio casos.json."
        for caso in casos:
            if caso["id"] == id_caso:
                return json.dumps(caso, ensure_ascii=False)
        return f"No se encontró ningún expediente con ID {id_caso}."
    except Exception as e:
        return f"Error al acceder a casos.json: {str(e)}"


@tool
def consultar_resoluciones_previas(id_caso: str) -> str:
    """Consulta la memoria de largo plazo (registro_resoluciones.json) para verificar si un expediente ya fue dictaminado. Entrada: solo el ID, ej: CASO-001."""
    id_caso = _extraer_id(id_caso)
    try:
        registros = _leer_json("registro_resoluciones.json", [])
        for r in registros:
            if r.get("id_caso") == id_caso:
                return (f"ALERTA DUPLICADO: El expediente {id_caso} YA fue dictaminado como: "
                        f"{r.get('veredicto')} (monto ${r.get('monto_aprobado')}). "
                        f"Justificación: {r.get('fundamento')}")
        return f"Expediente {id_caso} no registra liquidaciones previas. Procede su procesamiento."
    except Exception as e:
        return f"Error al consultar memoria de largo plazo: {str(e)}"


@tool
def consultar_normativa_rag(consulta: str) -> str:
    """Busca por similitud semántica los artículos de la póliza más relevantes. Entrada: una descripción corta del siniestro."""
    if vectorstore_poliza is None:
        return "Error: Base vectorial de póliza no disponible."
    try:
        docs = vectorstore_poliza.similarity_search(consulta, k=3)
        if not docs:
            return "No se encontraron cláusulas contractuales concordantes."
        return "\n---\n".join(d.page_content for d in docs)
    except Exception as e:
        return f"Error en la búsqueda semántica: {str(e)}"


PLAZO_DENUNCIA_DIAS = 10   # Artículo 4
TOPE_TERCEROS = 10000      # Artículo 3
_RE_ACCESORIOS = re.compile(r"\b(radio|estéreo|estereo|espejos?|accesorios?)\b")
_RE_NATURALEZA = re.compile(r"incendi|inundaci|terremoto|sismo|temporal")


def _regla_decision(caso: dict):
    texto = f"{caso.get('tipo', '')} {caso.get('descripcion', '')}".lower()
    monto = float(caso.get("monto_estimado", 0) or 0)
    dias = float(caso.get("dias_desde_siniestro", 0) or 0)

    # Prioridad 1: plazo de denuncia (Art. 4)
    if dias > PLAZO_DENUNCIA_DIAS:
        return ("Escalado", "Artículo 4", 0.0,
                f"Denuncia fuera de plazo ({dias:g} días; máximo {PLAZO_DENUNCIA_DIAS}). "
                "El Artículo 4 permite rechazarla; se deriva a revisor humano.")
    # Prioridad 2: exclusiones (Arts. 2 y 7)
    if _RE_ACCESORIOS.search(texto):
        return ("Denegado", "Artículo 2", 0.0,
                "Se trata de un robo parcial (accesorios removibles), no de un robo total del vehículo. "
                "La póliza solo cubre el robo total (Artículo 1) y excluye expresamente el robo de accesorios como radios y espejos (Artículo 2).")
    if _RE_NATURALEZA.search(texto):
        return ("Escalado", "Artículo 7", 0.0,
                "Incendio o fenómeno natural no cubierto salvo cobertura adicional; "
                "se deriva a revisor humano para verificar si existe.")
    # Prioridad 3: coberturas y topes (Arts. 1, 3 y 8)
    if "robo total" in texto or "sustracción completa" in texto:
        return ("Aprobado", "Artículo 1", monto,
                f"Se trata de un robo total del vehículo, cubierto por el Artículo 1. Monto aprobado ${monto:.0f}.")
    if "colisi" in texto or "choque" in texto:
        if monto <= TOPE_TERCEROS:
            return ("Aprobado", "Artículo 3", monto,
                    f"Se trata de una colisión cuyo monto (${monto:.0f}) está dentro del tope de ${TOPE_TERCEROS} del Artículo 3.")
        return ("Escalado", "Artículo 8", 0.0,
                f"Monto ${monto:.0f} supera el tope de${TOPE_TERCEROS} del Artículo 3; "
                "el Artículo 8 exige revisor humano.")
    return ("Escalado", "Artículo 8", 0.0,
            "Ningún artículo aplica claramente; se deriva a revisor humano.")


@tool
def evaluar_cobertura(id_caso: str) -> str:
    """Herramienta de RAZONAMIENTO: aplica a un expediente las reglas de la póliza (plazo, exclusiones, coberturas y topes) y devuelve la DECISIÓN VINCULANTE con la línea lista para registrar. Entrada: solo el ID, ej: CASO-001."""
    id_caso = _extraer_id(id_caso)
    try:
        casos = _leer_json("casos.json", [])
        caso = next((c for c in casos if c.get("id") == id_caso), None)
        if caso is None:
            return f"No se encontró ningún expediente con ID {id_caso}."
        veredicto, articulo, monto, just = _regla_decision(caso)
        return (
            f"DECISIÓN VINCULANTE para {id_caso}: {veredicto}, {articulo}, monto aprobado ${monto:.0f}. "
            f"Fundamento: {just}\n"
            "Usa EXACTAMENTE esta línea como Action Input de registrar_resolucion_final:\n"
            f"{id_caso} | {veredicto} | {articulo} | {monto:.0f} | {just}"
        )
    except Exception as e:
        return f"Error al evaluar la cobertura: {str(e)}"


def _buscar_registro(id_caso: str):
    try:
        return next((r for r in _leer_json("registro_resoluciones.json", []) if r.get("id_caso") == id_caso), None)
    except Exception:
        return None


def _registrar(id_caso, veredicto, articulo, monto, justificacion):
    try:
        registros = _leer_json("registro_resoluciones.json", [])
    except Exception:
        registros = []
    if any(r.get("id_caso") == id_caso for r in registros):
        return False, f"Error: El caso {id_caso} ya se encuentra registrado en el sistema."
    registros.append({
        "id_caso": id_caso,
        "veredicto": veredicto,
        "articulo_citado": articulo,
        "monto_aprobado": monto,
        "fundamento": justificacion,
        "fecha": datetime.now().isoformat(timespec="seconds"),
    })
    with open("registro_resoluciones.json", "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=4)
    return True, f"Éxito: Expediente {id_caso} archivado con veredicto '{veredicto}' y monto ${monto:.0f}."


@tool
def registrar_resolucion_final(datos: str) -> str:
    """Guarda el dictamen en 'registro_resoluciones.json' (no permite duplicados). Entrada: UNA sola línea con 5 campos separados por '|': ID_CASO | VEREDICTO | ARTICULO_CITADO | MONTO_APROBADO | JUSTIFICACION."""
    partes = [p.strip().strip("\"'") for p in str(datos).split("|", 4)]
    if len(partes) < 5:
        return ("Error de formato: se requieren 5 campos separados por '|': "
                "ID_CASO | VEREDICTO | ARTICULO_CITADO | MONTO_APROBADO | JUSTIFICACION")
    id_raw, veredicto_raw, articulo, monto_raw, justificacion = partes

    id_caso = _extraer_id(id_raw)
    v = veredicto_raw.lower()
    if v.startswith("aprob"):
        veredicto = "Aprobado"
    elif v.startswith("deneg") or v.startswith("rechaz"):
        veredicto = "Denegado"
    elif v.startswith("escal"):
        veredicto = "Escalado"
    else:
        return "Error: VEREDICTO debe ser Aprobado, Denegado o Escalado."
    try:
        monto = float(re.sub(r"[^\d.]", "", monto_raw) or 0)
    except ValueError:
        return "Error: MONTO_APROBADO debe ser un número (use 0 si no se aprueba monto)."

    return _registrar(id_caso, veredicto, articulo, monto, justificacion)[1]


def _texto_articulo(numero: int):
    try:
        with open("poliza.txt", "r", encoding="utf-8") as f:
            texto = f.read()
        for n, cuerpo in re.findall(
            r"^ARTÍCULO\s+(\d+)\s*:\s*(.+?)(?=^ARTÍCULO\s+\d+\s*:|\Z)", texto, re.MULTILINE | re.DOTALL
        ):
            if int(n) == numero:
                return cuerpo.strip()
    except Exception:
        pass
    return None


def _resolver_caso(id_caso: str):
    caso = _obtener_caso(id_caso)
    if caso is None:
        return None
    veredicto, articulo, monto, justificacion = _regla_decision(caso)
    _registrar(id_caso, veredicto, articulo, monto, justificacion)
    return _buscar_registro(id_caso)


def _componer_respuesta(reg: dict, recuperado: bool = False, nota=None) -> str:
    icono = {"Aprobado": "✅", "Denegado": "❌", "Escalado": "🔼"}.get(reg.get("veredicto"), "")
    lineas = [f"**{icono} Dictamen {reg['id_caso']}: {reg['veredicto']}**", "", reg.get("fundamento", "")]
    m = re.search(r"\d+", str(reg.get("articulo_citado", "")))
    texto_art = _texto_articulo(int(m.group(0))) if m else None
    if texto_art:
        lineas += ["", f"> **{reg['articulo_citado']}** de la póliza: {texto_art}"]
    lineas += ["", f"**Monto aprobado:** ${float(reg.get('monto_aprobado', 0)):,.0f}",
               "", "🗂️ Dictamen guardado en la memoria de largo plazo."]
    if nota:
        lineas += ["", nota]
    return "\n".join(lineas)


def _mensaje_duplicado(reg: dict) -> str:
    return (f"El expediente **{reg['id_caso']}** ya fue dictaminado el {reg.get('fecha', '(sin fecha)')}: "
            f"**{reg['veredicto']}** ({reg['articulo_citado']}), monto ${float(reg.get('monto_aprobado', 0)):,.0f}.\n\n"
            f"{reg.get('fundamento', '')}\n\n"
            "No se registra un nuevo dictamen. (Fuente: memoria de largo plazo)")


herramientas = [
    obtener_expediente_caso,
    consultar_resoluciones_previas,
    consultar_normativa_rag,
    evaluar_cobertura,
    registrar_resolucion_final,
]

# --- 5. CORE ENGINE Y PLANIFICACIÓN REACT (IE2, IE3, IE5) ---
llm = ChatOllama(model="hermes3", temperature=0.0)

# Memoria de corto plazo de LangChain (IE3)
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=False,
        k=4
    )

plantilla_agente = """Eres un Agente Liquidador de Siniestros corporativo. Respondes siempre en español.
Tienes acceso a las siguientes herramientas:
{tools}

REGLAS DE DECISIÓN Y PLANIFICACIÓN:
- Si el usuario solo saluda, hace una pregunta general o no menciona un expediente (ej: CASO-001), NO uses herramientas: responde con Final Answer presentándote y pidiendo el ID del caso.
- Si el usuario menciona un expediente, sigue estos pasos en orden, una herramienta por paso:
  1. obtener_expediente_caso con el ID (solo el ID, ej: CASO-002).
  2. consultar_resoluciones_previas con el ID. Si responde ALERTA DUPLICADO, NO registres nada y entrega el Final Answer.
  3. consultar_normativa_rag con una consulta que describa el tipo de siniestro.
  4. evaluar_cobertura con el ID. Devuelve la DECISIÓN VINCULANTE (veredicto, artículo y monto). NO la recalcules.
  5. registrar_resolucion_final usando EXACTAMENTE la línea que entregó evaluar_cobertura como Action Input.
  6. En cuanto recibas 'Éxito: ...', TU SIGUIENTE ACCIÓN DEBE SER DIRECTAMENTE 'Final Answer:' con el resumen final. NO ejecutes ninguna otra herramienta ni repitas pasos.
- Si una herramienta devuelve un error, explica el problema en el Final Answer.

Utiliza estrictamente el siguiente formato ReAct:

Question: Requerimiento del usuario
Thought: Reflexión sobre si debo usar una herramienta o responder directamente
Action: Nombre de la herramienta, exactamente una de [{tool_names}] (SOLO si es necesario usarla)
Action Input: Parámetro de entrada
Observation: Resultado de la herramienta
... (este ciclo Thought/Action/Action Input/Observation se repite si es necesario)
Thought: Ya tengo toda la información o no requiero más herramientas
Final Answer: Mensaje final dirigido al usuario

Historial previo:
{chat_history}

Pregunta: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=plantilla_agente,
    input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
)

agente = create_react_agent(llm=llm, tools=herramientas, prompt=prompt)
ejecutor = AgentExecutor(
    agent=agente,
    tools=herramientas,
    memory=st.session_state.agent_memory,
    verbose=True,
    handle_parsing_errors=(
        "Formato inválido. Responde SOLO con: 'Thought: ...' seguido de "
        "'Action: <herramienta>' y 'Action Input: <entrada>', "
        "o bien 'Thought: ...' seguido de 'Final Answer: <respuesta>'."
    ),
    max_iterations=10,
    return_intermediate_steps=True,
)

PROMPT_CONVERSACIONAL = """Eres un Agente Liquidador de Siniestros corporativo. Respondes en español, de forma breve y cordial.
Si el usuario saluda o hace una pregunta general, respóndele y pídele el ID del expediente a procesar (ej: CASO-001).
En este modo NO tienes herramientas: no has revisado ningún expediente, ni puedes enviar correos, tramitar, registrar o resolver nada.
Nunca afirmes ni prometas que lo harás. Si el usuario relata un siniestro sin dar un ID, solo expresa comprensión brevemente y pide el ID (ej: CASO-001).
No inventes datos de casos ni resoluciones.

Historial previo:
{historial}

Usuario: {entrada}
Respuesta:"""

PROMPT_CLASIFICAR = """Clasifica el relato de un cliente de seguros en UNA sola etiqueta:
robo total, robo parcial, colisión, incendio, otro.
Responde únicamente con la etiqueta.

Relato: {relato}
Etiqueta:"""

# --- 6. INTERFAZ STREAMLIT Y GESTIÓN CONVERSACIONAL ---
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "caso_pendiente" not in st.session_state:
    st.session_state.caso_pendiente = None

for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.write(_md(msg["content"]))

entrada_usuario = st.chat_input("Ingresa el ID del caso o instrucción (ej: Liquidar CASO-001)...")

if entrada_usuario:
    st.chat_message("user").write(_md(entrada_usuario))

    contexto_conversacion = st.session_state.agent_memory.load_memory_variables({}).get("chat_history", "")

    with st.chat_message("assistant"):
        with st.spinner("Analizando tu caso..."):
            try:
                historial_txt = contexto_conversacion if contexto_conversacion else "Sin contexto previo."
                pendiente = st.session_state.caso_pendiente
                id_detectado = _detectar_id(entrada_usuario)
                id_proceso, relato_proceso, intro, nota, ruta_txt = None, None, "", None, ""
                salida, pasos, texto_agente, recuperado = None, [], None, False

                # --- 1. Enrutamiento adaptativo (IE5) ---
                if id_detectado:
                    caso = _obtener_caso(id_detectado)
                    t_msg = _clasificar(entrada_usuario)
                    tipo_rel = t_msg or (pendiente.get("tipo") if pendiente else None)
                    relato_ref = entrada_usuario if t_msg else (pendiente["relato"] if pendiente else None)
                    ya_advertido = bool(pendiente and pendiente.get("discrepancia") == id_detectado)
                    if _hay_discrepancia(tipo_rel, caso) and not ya_advertido:
                        st.caption("🧭 Ruta: el ID no calza con tu relato → pide confirmación antes de procesar")
                        compat, _ = _ranking_por_tipo(relato_ref, tipo_rel)
                        mejor = next((c for c in compat if c["id"] != id_detectado), None)
                        st.session_state.caso_pendiente = {
                            "ids": [id_detectado], "relato": relato_ref,
                            "tipo": tipo_rel, "discrepancia": id_detectado,
                        }
                        salida = _mensaje_discrepancia(tipo_rel, caso, mejor)
                    else:
                        id_proceso, relato_proceso = id_detectado, relato_ref
                        ruta_txt = f"expediente {id_detectado} detectado"
                        if _hay_discrepancia(tipo_rel, caso):
                            nota = _nota_discrepancia(tipo_rel, caso)

                elif pendiente and pendiente.get("discrepancia") and _es_confirmacion(entrada_usuario):
                    id_proceso, relato_proceso = pendiente["ids"][0], pendiente["relato"]
                    ruta_txt = f"confirmación del usuario sobre {id_proceso}"
                    caso_conf = _obtener_caso(id_proceso)
                    if _hay_discrepancia(pendiente.get("tipo"), caso_conf):
                        nota = _nota_discrepancia(pendiente["tipo"], caso_conf)

                elif pendiente and pendiente.get("opciones") and _elegir_opcion(entrada_usuario, pendiente["opciones"]):
                    id_proceso = _elegir_opcion(entrada_usuario, pendiente["opciones"])
                    relato_proceso = pendiente["relato"]
                    intro = f"Perfecto, trabajaré con **{id_proceso}**.\n\n"
                    ruta_txt = f"el usuario eligió {id_proceso}"

                elif _parece_siniestro(entrada_usuario) or (
                    pendiente and pendiente.get("pregunta") and _RE_REFINA.search(entrada_usuario)
                ):
                    relato = f"{pendiente['relato']}. {entrada_usuario}" if pendiente else entrada_usuario
                    tipo = (_tipo_del_relato(entrada_usuario)
                            or (pendiente.get("tipo") if pendiente else None)
                            or _tipo_del_relato(relato))
                    if tipo == "robo" and re.search(r"\b(completo|entero|todo)\b", entrada_usuario, re.IGNORECASE):
                        tipo = "robo total"
                    compat, descartados = _ranking_por_tipo(relato, tipo)
                    previos = pendiente.get("intentos", 0) if pendiente else 0

                    if compat and tipo and (len(compat) == 1 or previos >= 1):
                        elegido = compat[0]
                        id_proceso, relato_proceso = elegido["id"], relato
                        ficha = f"**{elegido['id']}** ({elegido['tipo']}: {elegido['descripcion']})"
                        intro = ((f"No pude precisar más, así que asumí el caso más probable: {ficha}. "
                                  if len(compat) > 1 else
                                  f"Identifiqué tu caso como {ficha} a partir de tu relato. ")
                                 + "Si no corresponde, indícame el ID correcto.\n\n")
                        ruta_txt = "relato analizado → expediente identificado automáticamente"
                    elif compat and previos < 2:
                        pregunta = _pregunta_aclaratoria(tipo)
                        st.caption("🧭 Ruta: relato ambiguo → una pregunta para afinar (sin pedir el ID)")
                        st.session_state.caso_pendiente = {
                            "ids": [c["id"] for c in compat], "relato": relato, "tipo": tipo,
                            "pregunta": pregunta, "intentos": previos + 1,
                            "opciones": [c["id"] for c in compat] if tipo not in (None, "robo") else None,
                        }
                        salida = _mensaje_pregunta(tipo, compat, pregunta)
                    else:
                        st.caption("🧭 Ruta: relato sin coincidencias en los expedientes")
                        st.session_state.caso_pendiente = None
                        previo_txt = (f"Analicé tu relato: parece {_ETIQUETAS[tipo]}, pero no tengo expedientes de ese tipo. "
                                      if tipo else "No logré relacionar tu relato con ningún expediente registrado. ")
                        salida = previo_txt + "¿Puedes indicarme el ID de tu caso (ej: CASO-001)?"

                else:
                    if pendiente and pendiente.get("pregunta"):
                        st.caption("🧭 Ruta: respuesta no reconocida → repite la pregunta pendiente")
                        salida = "No logré entender tu respuesta. " + pendiente["pregunta"]
                    else:
                        st.caption("🧭 Ruta: sin expediente → respuesta directa del LLM (sin herramientas)")
                        salida = llm.invoke(
                            PROMPT_CONVERSACIONAL.format(historial=historial_txt, entrada=entrada_usuario)
                        ).content

                # --- 2. Ejecución ReAct sobre el caso identificado ---
                if id_proceso:
                    st.session_state.caso_pendiente = None
                    caso = _obtener_caso(id_proceso)
                    previo = _buscar_registro(id_proceso)
                    if caso is None:
                        st.caption("🧭 Ruta: expediente inexistente")
                        salida = f"No encontré ningún expediente con ID {id_proceso}. Verifica el ID e inténtalo de nuevo."
                    elif previo:
                        st.caption("🧭 Ruta: memoria de largo plazo → expediente ya dictaminado")
                        salida = intro + _mensaje_duplicado(previo)
                    else:
                        st.caption(f"🧭 Ruta: {ruta_txt} → ciclo ReAct con herramientas")
                        entrada_agente = (f"Liquidar {id_proceso}."
                                          + (f" Relato del cliente: {relato_proceso}" if relato_proceso else "")
                                          + f"\n(ID del expediente normalizado: {id_proceso})")
                        try:
                            res = ejecutor.invoke({"input": entrada_agente})
                            texto_agente = res.get("output", "")
                            pasos = res.get("intermediate_steps", [])
                        except Exception as e_agente:
                            texto_agente = f"(El agente falló: {e_agente})"

                        registro = _buscar_registro(id_proceso)
                        if registro is None:
                            registro = _resolver_caso(id_proceso)
                            recuperado = True
                        salida = intro + _componer_respuesta(registro, recuperado, nota)

                # Traza de decisiones ReAct (evidencia para la evaluación)
                if pasos or texto_agente:
                    esperadas = ["obtener_expediente_caso", "consultar_resoluciones_previas",
                                 "consultar_normativa_rag", "evaluar_cobertura", "registrar_resolucion_final"]
                    usadas = [a.tool for a, _ in pasos]
                    with st.expander(f"🔍 Traza del agente ({len(pasos)} pasos)"):
                        st.markdown("**Plan esperado:** " + " → ".join(
                            f"{'✅' if t in usadas else '⬜'} `{t}`" for t in esperadas))
                        for i, (accion, observacion) in enumerate(pasos, 1):
                            st.markdown(f"**{i}. `{accion.tool}`** ← `{str(accion.tool_input)[:200]}`")
                            st.caption(_md(str(observacion)[:500]))

                st.write(_md(salida))
                st.session_state.display_history.append({"role": "user", "content": entrada_usuario})
                st.session_state.display_history.append({"role": "assistant", "content": salida})
            except Exception as e:
                st.error(f"Error en ejecución: {str(e)}")

# --- BARRA LATERAL (MEMORIA A LARGO PLAZO) ---
with st.sidebar:
    st.header("📂 Memoria de Largo Plazo")
    try:
        registros_sidebar = _leer_json("registro_resoluciones.json", [])
    except Exception:
        registros_sidebar = []
    if registros_sidebar:
        st.json(registros_sidebar)
    else:
        st.write("Sin registros almacenados.")
    if st.button("🗑️ Borrar memoria de largo plazo"):
        if os.path.exists("registro_resoluciones.json"):
            os.remove("registro_resoluciones.json")
        st.rerun()
    if st.button("Reiniciar Sesión"):
        st.session_state.display_history = []
        st.session_state.caso_pendiente = None
        st.session_state.agent_memory.clear()
        st.rerun()