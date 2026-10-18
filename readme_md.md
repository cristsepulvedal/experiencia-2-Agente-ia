# ⚖️ Agente Autónomo de Liquidación de Siniestros Automotrices

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/LLM-Hermes%203%20%7C%20Ollama-purple.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)

**Asignatura:** ISY0101 – Ingeniería de Soluciones con IA  
**Evaluación:** Evaluación Parcial N°2 (Encargo y Presentación)  
**Institución:** Duoc UC – Escuela de Informática y Telecomunicaciones  

---

## 📋 Descripción del Proyecto

Este proyecto implementa un **Agente Autónomo de Liquidación de Siniestros** diseñado bajo el paradigma **ReAct (*Reasoning and Acting*)**. El agente asiste en el flujo de negocio de una aseguradora automotriz mediante la comprensión de lenguaje natural informal de los asegurados, la recuperación semántica de normativas contractuales (RAG), la evaluación determinista de coberturas y la persistencia de resoluciones en memoria de largo plazo.

### Capacidades Principales
- **Comprensión Semántica de Relatos:** Permite a los clientes describir siniestros en lenguaje cotidiano (ej: *"se robaron mi auto en el mall"*, *"choqué en la autopista"*), asociándolos automáticamente con el expediente correspondiente sin forzar códigos técnicos de entrada.
- **Flujo Cognitivo ReAct:** Razonamiento secuencial mediante el ciclo *Thought / Action / Action Input / Observation / Final Answer*.
- **Memoria Dual (C/P y L/P):**
  - **Corto Plazo:** Mantenimiento de ventana conversacional mediante `ConversationBufferWindowMemory` ($k=4$).
  - **Largo Plazo:** Persistencia de dictámenes en formato JSON estructurado con prevención activa de doble liquidación.
- **RAG Contractual con ChromaDB:** Indexación y recuperación por similitud semántica de artículos contractuales de la póliza automotriz utilizando embeddings densos `nomic-embed-text`.
- **Trazabilidad en Tiempo Real:** Inspección transparente del razonamiento y herramientas ejecutadas en la interfaz de Streamlit.

---

## 🏛️ Arquitectura del Sistema (IE8)

El siguiente esquema ilustra la orquestación e interacción entre los módulos del sistema:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CAPA DE USUARIO (Streamlit UI)                       │
│     - Entrada de Chat / Mensajes                                       │
│     - Visualizador Expandible de Traza ReAct                           │
│     - Panel Lateral de Auditoría (Memoria de Largo Plazo)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Texto informal / ID)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               MÓDULO DE ENRUTAMIENTO Y DESAMBIGUACIÓN                  │
│     - Extracción Regex de ID (CASO-XXX)                                │
│     - Clasificación Semántica (nomic-embed-text + ChromaDB)            │
│     - Detección de Discrepancias fácticas (Relato vs Expediente)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Expediente identificado)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               NÚCLEO AGÉNTICO ReAct (LangChain Engine)                 │
│  ┌──────────────────────┐               ┌───────────────────────────┐  │
│  │   LLM Core Engine    │ <-----------> │  Memoria de Corto Plazo   │  │
│  │   Hermes 3 (Ollama)  │               │  BufferWindowMemory (k=4) │  │
│  └──────────────────────┘               └───────────────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Invocación de Herramientas)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 CATÁLOGO DE HERRAMIENTAS MODULARES                     │
│  1. obtener_expediente_caso --------> casos.json (Repositorio)         │
│  2. consultar_resoluciones_previas -> registro_resoluciones.json (L/P) │
│  3. consultar_normativa_rag --------> ChromaDB / poliza.txt (RAG)      │
│  4. evaluar_cobertura --------------> Motor Determinista de Reglas     │
│  5. registrar_resolucion_final -----> registro_resoluciones.json (W)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Herramientas Integradas (IE1)

| Herramienta | Tipo | Entrada | Propósito Operacional |
| :--- | :---: | :---: | :--- |
| `obtener_expediente_caso` | Consulta | `str` (ID Caso) | Recupera los atributos del siniestro (tipo, descripción, monto estimado, días transcurridos) desde `casos.json`. |
| `consultar_resoluciones_previas` | Consulta / L/P | `str` (ID Caso) | Inspecciona `registro_resoluciones.json` para validar si el siniestro ya fue dictaminado y evitar duplicidad. |
| `consultar_normativa_rag` | Búsqueda Semántica | `str` (Consulta) | Ejecuta búsqueda por similitud de coseno en ChromaDB sobre las cláusulas de `poliza.txt`. |
| `evaluar_cobertura` | Razonamiento | `str` (ID Caso) | Aplica la jerarquía de reglas de la póliza (plazos, exclusiones, coberturas, topes) y genera el dictamen vinculante. |
| `registrar_resolucion_final` | Escritura | `str` (Línea estructurada) | Persiste el dictamen consolidado en `registro_resoluciones.json` con marca temporal para auditoría. |

---

## ⚙️ Esquema Jerárquico de Planificación (IE5)

La herramienta vinculante `evaluar_cobertura` ejecuta una lógica priorizada estricta para garantizar consistencia contractual:

1. **Prioridad 1 (Plazo de Denuncia - Art. 4):** Si `dias_desde_siniestro > 10` días corridos $\rightarrow$ **Escalado** (Revisor Humano).
2. **Prioridad 2 (Exclusiones Expresas - Art. 2):** Sustracción de accesorios removibles (radio, paneles, espejos) $\rightarrow$ **Denegado** (\$0).
3. **Prioridad 3 (Fenómenos Naturales - Art. 7):** Daños por incendio, inundación o sismos sin cobertura adicional $\rightarrow$ **Escalado**.
4. **Prioridad 4 (Cobertura Contractual - Art. 1):** Robo total del vehículo $\rightarrow$ **Aprobado** (Monto estimado total).
5. **Prioridad 5 (Topes de Responsabilidad - Arts. 3 y 8):**
   - Choque con monto $\le \$10,000$ USD $\rightarrow$ **Aprobado**.
   - Choque con monto $> \$10,000$ USD $\rightarrow$ **Escalado** (Supera límite contractual).

---

## 🚀 Requisitos e Instalación

### 1. Prerrequisitos
- **Python 3.10 o superior**
- **Ollama** instalado y en ejecución en el sistema.

### 2. Modelos en Ollama
Abre una terminal o CMD y descarga los modelos requeridos:

```cmd
# Descarga del modelo de lenguaje
ollama pull hermes3

# Descarga del modelo de embeddings semánticos
ollama pull nomic-embed-text
```

Asegúrate de que el servicio de Ollama esté en ejecución:
```cmd
ollama serve
```

### 3. Clonar Repositorio y Configurar Entorno Virtual

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/agente-liquidacion-siniestros.git
cd agente-liquidacion-siniestros

# Crear y activar entorno virtual (Windows)
python -m venv venv
venv\Scripts\activate

# Crear y activar entorno virtual (Linux/macOS)
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependencias

Crea o utiliza el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

*Contenido de `requirements.txt`:*
```text
streamlit>=1.35.0
langchain>=0.1.20
langchain-community>=0.0.38
langchain-core>=0.1.52
chromadb>=0.4.24
```

---

## 💻 Ejecución de la Aplicación

Para iniciar el agente y abrir el dashboard en el navegador:

```bash
streamlit run asistente_final_5.py
```

La aplicación quedará accesible en: `http://localhost:8501`.

---

## 🧪 Casos de Prueba para Demostración (IE7)

| Escenario | Entrada del Usuario | Comportamiento del Agente | Veredicto Esperado |
| :--- | :--- | :--- | :--- |
| **Robo Parcial (Exclusión)** | *"Me robaron la radio del auto en el estacionamiento"* | Clasifica el siniestro, asocia `CASO-001`, consulta RAG y aplica exclusión del Art. 2. | **Denegado** (\$0) |
| **Robo Total (Cubierto)** | *"Se robaron mi auto en el mall con violencia"* | Asocia `CASO-002`, valida cobertura del Art. 1 y aprueba monto. | **Aprobado** (\$12,000) |
| **Colisión dentro de Tope** | *"Choque por alcance en la autopista"* | Asocia `CASO-003`, valida monto bajo tope del Art. 3. | **Aprobado** (\$7,500) |
| **Colisión sobre Tope** | Ingrese `CASO-004` | Detecta monto de \$16,000 USD (supera tope de \$10,000 del Art. 3) y deriva según Art. 8. | **Escalado** (Humano) |
| **Incendio / Fenómeno** | *"Se me quemó el auto en el estacionamiento"* | Asocia `CASO-005` y deriva según Art. 7. | **Escalado** (Humano) |
| **Duplicado en Memoria L/P** | Ingrese nuevamente un caso resuelto (ej: `CASO-001`) | Detecta registro en `registro_resoluciones.json`, alerta duplicado y no reescribe. | **Alerta Duplicado** |

---

## 📂 Estructura del Repositorio

```text
├── asistente_final_5.py         # Aplicación principal Streamlit y orquestación ReAct
├── casos.json                   # Base de datos simulada de expedientes reportados
├── poliza.txt                   # Documento contractual de póliza automotriz
├── registro_resoluciones.json   # Base de datos persistente (Memoria de Largo Plazo)
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación técnica de arquitectura y ejecución
```

---

## 📚 Referencias Bibliográficas (IE9)

- Chase, H. (2023). *LangChain: Building applications with LLMs through composability and modularity*. LangChain Inc. https://python.langchain.com
- Chroma Authors. (2023). *Chroma: The open-source embedding database*. Chroma Inc. https://www.trychroma.com
- Nomic AI. (2024). *Nomic Embed: High-dimensional open-weights text embeddings*. Nomic AI. https://www.nomic.ai
- Ollama Community. (2024). *Ollama: Get up and running with large language models locally*. https://ollama.com
- Streamlit Inc. (2024). *Streamlit: The fastest way to build and share data apps*. Snowflake Inc. https://streamlit.io
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). *ReAct: Synergizing reasoning and acting in language models*. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/2210.03629