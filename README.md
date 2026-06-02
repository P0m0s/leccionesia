# Estimador CAG (FastAPI + Streamlit)

API REST que genera **estimaciones de software** con un LLM usando **CAG** (contexto estático): las instrucciones y ejemplos few-shot viven en plantillas **Jinja2** versionadas (`v1`, `v2`, y la familia `chat/v1` / `chat/v2` para conversaciones), no en código Python disperso. El servicio admite **OpenAI** o **Anthropic** según configuración, **caché exact-match** sobre la petición serializada y **respuesta en streaming**.

A partir de la sesión 05 incorpora **memoria conversacional**, **adjuntos** (PDF / DOCX / TXT) y un **`project_metadata`** que se inyecta dinámicamente en el system prompt y se va enriqueciendo turno a turno.

**Mejoras adicionales sobre esa base (este sprint):**

| Categoría | Mejora |
|-----------|--------|
| Calidad | Salida **JSON estructurada** (Pydantic) con `line_items`, `phases`, `assumptions`, `risks`, `confidence`. |
| Calidad | **Calibración T-shirt sizing** (XS–XL) inyectada en el system prompt. |
| Calidad | **Few-shot dinámico** por `project_type` (`mobile_app`, `web_saas`, `internal_tool`, `data_pipeline`). |
| Calidad | Bandera `?refine=true` para una **pasada de auto-crítica** antes de devolver la estimación. |
| Calidad | Extractor LLM opcional de `project_metadata` y **resumen automático** al rotar la ventana deslizante. |
| Funcional | **Streaming NDJSON** para el chat conversacional (`/sessions/{id}/estimate/stream`). |
| Funcional | **Caché SHA-256** y **límites duros** para adjuntos (tamaño y cantidad). |
| Funcional | **Vision** (PNG/JPEG/GIF/WEBP) por Camino A multimodal. |
| Funcional | `GET /sessions/{id}` para **rehidratar** historial + metadata + métricas. |
| Funcional | **Comandos `/`** en el chat: `/reset`, `/metadata`, `/regenerate`, `/help`. |
| Funcional | Versión `chat/v2` con filosofía **adversarial** (cuestiona supuestos). |
| Funcional | Exportar conversación a **Markdown / PDF**. |
| UX | Tarjetas con iconos para `project_metadata`, indicador de ventana deslizante, render de tablas como `st.dataframe`, preview de adjuntos antes de enviar, system prompt efectivo en sidebar, theming corporativo. |
| Observ. | **Métricas por sesión** (tokens, llamadas LLM, coste estimado USD, latencia). |
| Observ. | **Self-confidence score** del modelo en cada turno. |
| Observ. | **Eval set** de briefs canónicos en `app/eval/`. |
| Robustez | **TTL + GC** de sesiones inactivas, **CI GitHub Actions** con tests y `ruff`. |

La interfaz **Streamlit** envía el mismo contrato HTTP que la API (formulario tipado + historial + streaming + pestaña conversacional con adjuntos y streaming), sin duplicar la lógica de prompts ni de proveedores en el cliente.

## Requisitos

- Python 3.11 o superior
- Cuenta **OpenAI** o **Anthropic** y una API key del proveedor que vayas a usar
- Opcional: [uv](https://docs.astral.sh/uv/) para gestionar el entorno

## Configuración

### 1. Variables de entorno

Copia el ejemplo y edítalo:

```bash
cp .env.example .env
```

En **Windows (PowerShell)** puedes usar:

```powershell
Copy-Item .env.example .env
```

| Variable | Descripción |
|----------|-------------|
| `LLM_PROVIDER` | `openai` o `anthropic` (define qué cliente usa el backend). |
| `OPENAI_API_KEY` | Obligatoria si `LLM_PROVIDER=openai`. |
| `ANTHROPIC_API_KEY` | Obligatoria si `LLM_PROVIDER=anthropic`. |
| `OPENAI_MODEL` | ID del modelo OpenAI (por defecto `gpt-4o-mini`). |
| `ANTHROPIC_MODEL` | ID del modelo Anthropic (por defecto `claude-3-5-haiku-20241022`). |
| `ENABLE_AUTO_SUMMARY` | `true`/`false`. Llama al LLM al rotar la ventana para mantener un resumen del histórico. Por defecto `true`. |
| `ENABLE_LLM_METADATA` | `true`/`false`. Tras cada turno hace una llamada extra al LLM para enriquecer `project_metadata`. Por defecto `false` (la heurística regex sigue activa siempre). |
| `SESSION_TTL_SECONDS` | TTL de inactividad antes de descartar una sesión. Por defecto 24 h. |
| `EMBEDDING_BACKEND` | `openai` o `local` para el pipeline de búsqueda semántica. |
| `EMBEDDING_MODEL` | Modelo OpenAI de embeddings (por defecto `text-embedding-3-small`). |
| `EMBEDDING_BATCH_SIZE` | Lote para `embed_many` (por defecto 100). |
| `VECTOR_STORE_PATH` | JSON opcional para cargar/guardar el vector store entre reinicios. |

Solo necesitas rellenar la clave del proveedor que elijas; la otra puede quedar vacía.

### 2. Streamlit apuntando a la API

Streamlit llama por HTTP a la API. Por defecto usa `http://127.0.0.1:8000`. Si tu servidor corre en otro host o puerto, define:

```bash
export ESTIMATOR_API_BASE="http://127.0.0.1:8080"
```

En PowerShell:

```powershell
$env:ESTIMATOR_API_BASE = "http://127.0.0.1:8080"
```

En **Streamlit Cloud** u otro despliegue, puedes fijar `ESTIMATOR_API_BASE` en *Secrets* junto con las mismas claves que uses en local si quisieras llamar a una API desplegada (el flujo recomendado en curso es: API con las keys en el servidor y Streamlit solo con la URL pública).

## Instalación

### Con uv

```bash
uv sync
# opcional: dependencias de desarrollo (pytest)
uv sync --extra dev
```

### Con venv y pip

```bash
py -m venv .venv
.\.venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
```

## Arranque

**Terminal 1 — API**

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

O con el venv activado:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Documentación interactiva: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Salud: `GET http://127.0.0.1:8000/health`

## Pipeline de embeddings y búsqueda semántica (pre-session-06)

Pipeline mínimo para indexar documentos y recuperar fragmentos por similitud coseno:

1. **Chunking** (`chunking.py`) — trocea el texto con `tiktoken` (`cl100k_base`) respetando `max_tokens` (por defecto 300) sin partir palabras. Cada chunk lleva metadatos `chunk_id`, `start_char`, `end_char`.
2. **Embeddings** (`embeddings_service.py`) — `EmbeddingService` expone `embed_text` y `embed_many` (con batching). Por defecto usa OpenAI `text-embedding-3-small`; en tests o sin API puedes usar `EMBEDDING_BACKEND=local` (vectores deterministas).
3. **Vector store** (`vector_store.py`) — `InMemoryVectorStore` guarda `{embedding, metadata}` en memoria y busca por **similitud coseno**. Persistencia opcional en JSON (`save` / `load`).

### Indexar documentos

**HTTP — `POST /embed`**

```json
{ "text": "contenido del documento" }
```

Respuesta:

```json
{ "chunks_indexed": 3, "store_size": 3 }
```

**CLI**

```bash
uv run python -m cli.index --file ruta.txt
# opcional: persistir en disco
uv run python -m cli.index --file ruta.txt --persist data/vector_store.json
```

Si defines `VECTOR_STORE_PATH` en `.env`, la API carga ese JSON al arrancar.

### Buscar

**`POST /search`**

```json
{ "query": "texto de búsqueda", "k": 5 }
```

Respuesta:

```json
{
  "results": [
    {
      "score": 0.87,
      "chunk": "fragmento recuperado…",
      "metadata": { "chunk_id": 0, "start_char": 0, "end_char": 42 }
    }
  ]
}
```

Ejemplo con curl:

```bash
curl -s -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"FastAPI con PostgreSQL y migraciones Alembic."}'

curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"API Python FastAPI","k":3}'
```

### Limitaciones del vector store

- **En memoria**: un solo proceso; reiniciar el servidor vacía el store salvo que uses `VECTOR_STORE_PATH`.
- **Sin filtros**: no hay metadata filtering ni namespaces; todos los chunks comparten la misma lista.
- **Sin deduplicación**: volver a llamar `/embed` añade chunks nuevos (no reemplaza por documento).
- **Escala**: adecuado para demos y pruebas; producción requeriría un store externo (pgvector, Pinecone, etc.).

### Cambiar el modelo de embeddings

| Variable | Descripción |
|----------|-------------|
| `EMBEDDING_BACKEND` | `openai` (por defecto) o `local` (sin API). |
| `EMBEDDING_MODEL` | ID del modelo OpenAI (p. ej. `text-embedding-3-small`, `text-embedding-3-large`). |
| `EMBEDDING_BATCH_SIZE` | Tamaño de lote para `embed_many` (por defecto 100). |
| `OPENAI_API_KEY` | Obligatoria si `EMBEDDING_BACKEND=openai`. |

**Terminal 2 — Interfaz Streamlit** (con la API ya levantada)

```bash
uv run streamlit run streamlit_app.py
```

O:

```bash
python -m streamlit run streamlit_app.py
```

En la app verás el **formulario de producto** (POST JSON), opción de **streaming**, pestaña de **atajo por transcripción** e **historial** de respuestas. En la barra lateral se muestra una vista previa del system/user renderizado por Jinja para una petición de ejemplo.

## Uso de la API

### Cuerpo de la petición (`EstimationRequest`)

Todos los campos van en JSON salvo `prompt_version`, que es un **query parameter** opcional.

| Campo | Tipo | Notas |
|-------|------|--------|
| `description` | string | Entre **20** y **2000** caracteres. |
| `project_type` | enum | `mobile_app`, `web_saas`, `internal_tool`, `data_pipeline`. |
| `detail_level` | enum | `summary`, `medium`, `detailed`. |
| `output_format` | enum | `phases_table`, `line_items`, `narrative`. |
| `reference_projects` | lista de strings, opcional | Usada en plantillas **v2** (comparación de patrones). |

### `POST /estimate`

Estimación completa en un solo JSON de respuesta (`EstimationResponse`: `text`, `prompt_version`).

Query opcional: `prompt_version=v1` (por defecto) o `prompt_version=v2`.

**Ejemplo con curl (bash):**

```bash
curl -s -X POST "http://127.0.0.1:8000/estimate?prompt_version=v1" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Portal interno de aprobación de gastos con flujo multi-nivel y exportación contable.",
    "project_type": "internal_tool",
    "detail_level": "medium",
    "output_format": "line_items"
  }'
```

**PowerShell (`Invoke-RestMethod`):**

```powershell
$body = @{
  description = "Portal interno de aprobación de gastos con flujo multi-nivel y exportación contable."
  project_type = "internal_tool"
  detail_level = "medium"
  output_format = "line_items"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/estimate?prompt_version=v1" `
  -ContentType "application/json" -Body $body
```

### `POST /estimate/stream`

Misma petición que arriba; la respuesta es **texto plano** (`text/plain`) con el contenido generado por chunks (ideal para consumo desde Streamlit o clientes que quieran mostrar tokens al vuelo).

```bash
curl -N -X POST "http://127.0.0.1:8000/estimate/stream?prompt_version=v1" \
  -H "Content-Type: application/json" \
  -d '{"description":"MVP de reservas con notificaciones y panel admin mínimo.","project_type":"mobile_app","detail_level":"summary","output_format":"narrative"}'
```

### Sesiones conversacionales (memoria + adjuntos)

Pensadas para iterar varias rondas con el mismo proyecto. El servidor mantiene un **historial con ventana deslizante** (`MAX_TURNS = 6`) y un **`project_metadata`** acumulativo (nombre del proyecto, tamaño de equipo asumido, tecnologías mencionadas y alcance acordado) que se **re-inyecta en el system prompt en cada turno** vía Jinja2 (`app/prompts/chat/v1/system.j2`).

#### `POST /sessions`

Crea una sesión nueva en memoria. Devuelve `{"session_id": "<uuid4>"}`.

```bash
curl -s -X POST http://127.0.0.1:8000/sessions
```

#### `POST /sessions/{session_id}/estimate`

Recibe **`multipart/form-data`**:

| Campo | Tipo | Notas |
|-------|------|-------|
| `transcript` | string (form) | Mensaje del usuario. Vacío permitido si hay adjunto. |
| `attachments` | files (opcional) | Uno o varios PDF / DOCX / TXT / MD / **imágenes** (PNG, JPEG, GIF, WEBP). |

Query opcionales:

| Param | Tipo | Notas |
|-------|------|-------|
| `prompt_version` | `v1` / `v2` | Selecciona el system prompt del chat. `v2` es **adversarial**. |
| `refine` | bool | Si `true`, hace una segunda pasada de **auto-crítica** antes de responder. |

Devuelve un JSON con:

- `text`: resumen markdown (igual a `structured.summary_markdown`).
- `structured`: estimación tipada (`line_items`, `phases`, `assumptions`, `risks`, `total_hours_min/max`, `confidence`, `next_step`).
- `structured_ok`: `true` si el LLM devolvió JSON válido (si no, `text` mantiene la respuesta cruda).
- `refined`: `true` si se ejecutó la auto-crítica.
- `project_metadata`: incluido `conversation_summary` cuando ha habido overflow.
- `metrics`: acumulado de la sesión (`turns_count`, `llm_calls`, tokens, `estimated_cost_usd`).
- `attachments_processed` / `attachments_failed`.

```bash
curl -s -X POST "http://127.0.0.1:8000/sessions/$SID/estimate?prompt_version=v2&refine=true" \
  -F "transcript=El proyecto se llama Aurora. Vamos a usar Python y FastAPI." \
  -F "attachments=@spec.pdf;type=application/pdf"
```

##### Comandos `/`

Si `transcript` empieza por `/`, **no se llama al LLM**:

| Comando | Acción |
|---------|--------|
| `/help` | Lista todos los comandos. |
| `/reset` (alias `/clear`) | Vacía historial y metadata, mantiene `session_id`. |
| `/metadata` | Muestra el `project_metadata` actual. |
| `/regenerate` | Descarta la última respuesta y vuelve a llamar al LLM con el mismo turno. |

#### `POST /sessions/{session_id}/estimate/stream`

Mismo body que arriba. Devuelve **NDJSON** (`application/x-ndjson`):

```json
{"type": "start", "session_id": "…", "attachments_processed": ["a.pdf"], "attachments_failed": []}
{"type": "token", "delta": "## Plan v1\n"}
{"type": "token", "delta": "- Auth (S, 16–40h)\n"}
{"type": "final", "text": "## Plan v1…", "structured": {...}, "structured_ok": true, "project_metadata": {...}, "metrics": {...}}
```

#### `GET /sessions/{session_id}`

Rehidrata historial, `project_metadata` y métricas de la sesión.

##### Decisiones de diseño

- **Adjuntos de texto: Camino B (extracción local).** El servidor extrae texto con `pypdf` (PDF) o `python-docx` (DOCX) y lo concatena al transcript con el separador `--- attachment: <nombre> ---`. Compatible 1:1 con OpenAI y Anthropic.
- **Adjuntos imagen: Camino A (multimodal).** PNG/JPEG/GIF/WEBP se codifican en base64 e **inyectan** en el último user message como contenido multimodal en el formato propio de cada proveedor (`image_url` para OpenAI, `image` block para Anthropic). Ver `app/services/llm_service.py::_inject_images_*`.
- **Caché de adjuntos por SHA-256.** El mismo binario re-subido en turnos posteriores se sirve desde caché in-process (`app/services/attachments.py::_TEXT_CACHE`).
- **Límites duros de adjuntos.** Máximo `MAX_ATTACHMENT_BYTES = 10 MB` por archivo y `MAX_ATTACHMENTS_PER_TURN = 5` por turno.
- **Extracción de `project_metadata`:** **doble vía**:
  1. **Heurística regex** (siempre activa): catálogo cerrado de tecnologías + patrones para nombre de proyecto, tipo (`mobile_app`/`web_saas`/`internal_tool`/`data_pipeline`), tamaño de equipo y alcance.
  2. **Extractor LLM** (opt-in vía `ENABLE_LLM_METADATA=true`): tras cada turno hace una segunda llamada ligera que devuelve un JSON con campos refinados; se **fusiona** con la heurística sin pisar lo ya acordado.
- **Resumen al rotar ventana** (`ENABLE_AUTO_SUMMARY=true` por defecto): cuando un par user+assistant cae de la ventana, se llama al LLM con un summarizer ligero para acumular `conversation_summary` en `ProjectMetadata`. Si la llamada falla, el resumen previo se conserva.
- **Few-shot dinámico:** `app/prompts/chat/v1/examples/{project_type}.j2`. La versión `v2` reutiliza los mismos ejemplos por fallback.
- **Salida estructurada:** el system prompt instruye al LLM a devolver un JSON con un schema fijo. Si el LLM devuelve markdown libre, `parse_structured_response` cae a un fallback que copia todo a `summary_markdown` sin romper el contrato.
- **Estado.** Diccionario global en memoria (`SessionStore`); sin BBDD ni Redis. Una tarea de fondo (`_session_gc_loop` en `app/main.py`) limpia cada minuto las sesiones inactivas según `SESSION_TTL_SECONDS`.

### Caché y versiones de prompt

- **Caché exact-match** sobre `/estimate` (one-shot). Las sesiones conversacionales no se cachean porque cada turno depende del historial.
- **Versiones:** `v1` y `v2` viven en `app/prompts/estimation/`; los system prompts conversacionales están en `app/prompts/chat/v1/` (estándar) y `app/prompts/chat/v2/` (adversarial). Para añadir una versión nueva basta con crear la carpeta y registrar la versión en `_VALID_CHAT_VERSIONS`.

## Tests

Sin llamadas a APIs externas: los tests de sesiones mockean el wrapper LLM con `monkeypatch`, y los tests de prompts solo renderizan Jinja2.

```bash
uv run pytest tests/ -v
```

O con el venv activado:

```bash
.\.venv\Scripts\python -m pytest tests/ -v
```

Cobertura incluida (74 tests, < 1 s):

| Test | Qué cubre |
|------|-----------|
| `tests/prompts/test_estimation_v1.py` | Render de plantillas `estimation/v1`. |
| `tests/test_sessions_metadata.py` | Dos turnos → `project_metadata` cambia. |
| `tests/test_sessions_attachments.py` | PDF como adjunto cambia la estimación. |
| `tests/test_sessions_sliding_window.py` | 8 turnos → nunca supera `MAX_TURNS`. |
| `tests/test_structured_output.py` | Parseo del JSON estructurado + fallbacks. |
| `tests/test_fewshot_dynamic.py` | Detector de `project_type` y bloque de ejemplos. |
| `tests/test_sessions_stream.py` | `/estimate/stream` emite `start` / `token` / `final`; `GET /sessions/{id}`. |
| `tests/test_attachments_cache_and_limits.py` | Caché SHA-256, límites de tamaño y cantidad. |
| `tests/test_slash_commands.py` | `/reset`, `/metadata`, `/regenerate`, comando desconocido. |
| `tests/test_vision_attachments.py` | Detección de imagen + inyección multimodal (OpenAI/Anthropic). |
| `tests/test_refine.py` | `?refine=true` dispara segunda llamada. |
| `tests/test_metadata_llm_and_summary.py` | Extractor LLM + resumen automático al overflow. |
| `tests/test_chat_v2.py` | Versión adversarial del prompt. |
| `tests/test_session_ttl.py` | `evict_inactive` borra solo las antiguas. |
| `tests/test_session_metrics.py` | Acumulado de tokens y coste estimado. |
| `tests/test_eval_set.py` | Eval set canónico ejecutable con LLM mockeado. |
| `tests/test_frontend_utils.py` | Tarjetas, parseo de tablas markdown y export Markdown/PDF. |

Los tests asíncronos usan `httpx.AsyncClient` sobre el ASGI de FastAPI (sin levantar uvicorn).

### Lint

```bash
.\.venv\Scripts\python -m ruff check app/ tests/ streamlit_app.py
```

Configuración en `pyproject.toml` (selección `E F I B UP W` + ignores razonables).

### CI

GitHub Actions (`.github/workflows/ci.yml`) ejecuta los tests en Python 3.11 y 3.12 + `ruff check` en cada push y PR.

### Eval set

```bash
.\.venv\Scripts\python -m app.eval.runner --out eval-report.json
```

Briefs canónicos en `app/eval/briefs.py`; el runner verifica `project_type` detectado, tecnologías esperadas, temas requeridos, totales dentro de rango y `confidence`. **Requiere LLM real** (las claves del `.env`).

## Estructura del proyecto (resumen)

| Ruta | Rol |
|------|-----|
| `app/main.py` | FastAPI + `lifespan` con GC de sesiones inactivas. |
| `app/config.py` | Ajustes desde entorno (`pydantic-settings`). |
| `app/schemas.py` | Modelos Pydantic de entrada y salida. |
| `app/structured.py` | `StructuredEstimation` y parseo tolerante del JSON del LLM. |
| `app/sessions.py` | `ConversationHistory`, `ProjectMetadata`, `Session`, `SessionMetrics`, `SessionStore`. |
| `app/routers/estimations.py` | `POST /estimate` y `POST /estimate/stream`. |
| `app/routers/sessions.py` | Endpoints conversacionales (sync, streaming, GET). |
| `app/prompts/loader.py` | Render de plantillas one-shot y de chat con few-shot dinámico. |
| `app/prompts/estimation/v1/` … `v2/` | Plantillas Jinja2 one-shot. |
| `app/prompts/chat/v1/` | System prompt estándar + `examples/` por dominio + `refine.j2`. |
| `app/prompts/chat/v2/` | Versión adversarial. |
| `app/services/llm_service.py` | Wrapper multi-proveedor + streaming + inyección multimodal. |
| `app/services/estimate_service.py` | Orquestación one-shot: render, caché, llamada al LLM. |
| `app/services/session_service.py` | Orquestación por turno (adjuntos → historial → LLM → metadata → métricas). |
| `app/services/attachments.py` | Extracción local de texto + caché SHA-256 + codificación de imágenes. |
| `app/services/metadata_extractor.py` | Heurística regex para `ProjectMetadata` (incluye `project_type`). |
| `app/services/metadata_llm.py` | Extractor LLM + summarizer del overflow. |
| `app/services/slash_commands.py` | Dispatcher de comandos `/`. |
| `app/services/cost.py` | Tabla de precios por modelo (USD por 1K tokens). |
| `app/tiers/` | Patrón **tier**: `CallerContext`, `TIER_CONFIG`, schemas y pipelines por perfil. |
| `app/prompts/tiers/` | Templates Jinja2 por tier + parciales compartidos. |
| `app/eval/` | Eval set de briefs canónicos. |
| `app/frontend_utils.py` | Tarjetas, parseo de tablas markdown y export Markdown/PDF. |
| `.streamlit/config.toml` | Theming corporativo. |
| `streamlit_app.py` | Pestañas: conversación con adjuntos/streaming, formulario, transcripción, historial, sidebar con métricas y `system` prompt efectivo. |
| `tests/` | Tests async sin red, con LLM mockeado. |
| `.github/workflows/ci.yml` | Tests + ruff en GitHub Actions. |

El archivo `app/context/examples.py` es legado del primer ejercicio; los ejemplos few-shot activos están en los `.j2` dentro de `app/prompts/`.

## Patrón tier — perfiles adaptativos de cliente

Implementación del ejercicio *Prompts adaptativos por perfil de usuario* (módulo AI Engineering 2026/04). Una misma sesión conversacional puede responder al mismo brief con estructuras, vocabulario y pipelines **distintos** según el perfil del receptor.

### Qué es un tier

Un tier es una etiqueta sobre el `caller` que selecciona simultáneamente:

| Tier | Pipeline | Schema Pydantic | Template Jinja2 | Audiencia |
|------|----------|-----------------|------------------|-----------|
| `developer` | `single_call` | `DeveloperEstimate` | `tiers/developer.j2` | Ingenieros — componentes, riesgos técnicos, stack, drivers de incertidumbre. |
| `pm` | `single_call` | `PmEstimate` | `tiers/pm.j2` | PMs — fases, hitos, composición de equipo, blockers. |
| `executive` | `single_call` | `ExecutiveEstimate` | `tiers/executive.j2` | C-level — coste/duración headline, top-3 riesgos, go/no-go. |
| `research` | `deep_research` | `ResearchEstimate` | `tiers/research.j2` | Informe profundo con índice, secciones, citas y metodología. |

La fuente de verdad única está en `app/tiers/config.py::TIER_CONFIG`. Añadir un tier nuevo = una entrada en ese dict + un template + un schema.

### Cómo se propaga el tier

El backend **nunca** confía en un campo "suelto" en el body. La identidad llega siempre como un `CallerContext` construido por una dependency de FastAPI (`Depends(get_caller_context)`):

```python
class CallerContext(BaseModel):
    user_id: str
    tier: Literal["developer", "pm", "executive", "research"]
```

Hay dos estrategias canónicas, implementadas en `app/tiers/context.py`:

- **Opción B (activa)** — Headers simples en red privada. La dependency lee `X-Estimator-Tier` y `X-Estimator-User`. Apta para MVP / red de confianza. Si el header no llega, la dependency devuelve `None` y el endpoint cae al **flujo conversacional clásico** (compat hacia atrás).
- **Opción A (comentada)** — JWT firmado. Lista para activar: descomentar `get_caller_context_jwt`, instalar `python-jose[cryptography]` y configurar `ESTIMATOR_JWT_SECRET`. El frontend o el API gateway emite tokens con `sub` y `tier` como claims.

> **Nota sobre el frontend del MVP**: como aún no hay sistema de auth, el dropdown del Streamlit envía el tier directamente en el header. En producción se reemplaza la dependency por la versión JWT sin tocar el endpoint ni los pipelines.

### Cómo funciona cada pipeline

Definidas en `app/tiers/pipelines.py`:

- **`single_call`**: render del template tier → construir `[system, ...history]` → 1 llamada al LLM → validar contra el schema Pydantic del tier → añadir al historial. Memoria conversacional preservada igual que el flujo clásico.
- **`deep_research`**: misma estructura pero pensada para `o3-deep-research` con `web_search` y `code_interpreter` en background. Mientras no haya acceso a esa API, hace la llamada al provider configurado por defecto usando el template de investigación, que ya pide informe extenso con índice, secciones, citas y metodología.

`TIER_CONFIG["research"]` marca `background: true`, `estimated_latency_seconds: 600` y `estimated_cost_per_call_eur: 5.0` para que el frontend pueda advertir al usuario antes de disparar la pipeline.

### Cómo se usa desde el frontend

En la pestaña de Conversación hay un selector **Perfil del cliente (tier)** con 5 opciones:

1. `💬 Conversacional (sin tier)` — no envía header → flujo clásico (`StructuredEstimation`).
2. `🛠️ Developer` — envía `X-Estimator-Tier: developer`.
3. `🧭 PM` — `pm`.
4. `💼 Executive` — `executive`.
5. `🔬 Research` — `research` (lento/caro, deshabilita streaming).

Al cambiar de tier dentro de una misma sesión, **el historial se conserva**: puedes pedir lo mismo desde el punto de vista de un PM y después como executive para comparar.

Cada tier renderiza diferente en Streamlit:

- **Developer**: 3 métricas de horas, tabla de componentes, riesgos técnicos vs drivers de incertidumbre, supuestos de stack.
- **PM**: métricas de semanas/roles, tabla de fases, tabla de hitos, composición del equipo, blockers con icono de severidad.
- **Executive**: 3 metricas grandes (coste, duración, recomendación con badge), confianza, rationale, top-3 riesgos.
- **Research**: cabecera con totales, índice plegable, cada sección en su propio expander con citas, metodología, siguientes pasos.

### Cómo añadir un tier nuevo

1. Define un schema en `app/tiers/schemas.py` con la estructura específica.
2. Crea `app/prompts/tiers/<nombre>.j2`. Reutiliza `{% include "tiers/_project_metadata.j2" %}` y `{% include "tiers/_reference_estimates.j2" %}` para mantener la disciplina de parciales.
3. Añade la entrada a `TIER_CONFIG` con `pipeline`, `template`, `schema` y `model`. Si necesita una pipeline nueva, regístrala en `PIPELINE_HANDLERS`.
4. Añade el tier al `Literal` en `CallerContext.tier` y al `frozenset` `_VALID_TIERS` de `app/tiers/context.py`.
5. Frontend: añade una entrada a `TIER_OPTIONS` y, si quieres render rico, un `_render_tier_<nombre>` en `streamlit_app.py`.

### Anti-patrones evitados

- ❌ **Tier desde frontend sin verificación**: el backend valida vía dependency; en el MVP el dropdown envía un header pero la migración a JWT es trivial.
- ❌ **Schema único con branching**: 4 schemas Pydantic genuinamente distintos (`components` solo en Developer; `phases` solo en PM; `headline_cost_range` solo en Executive; `sections` solo en Research).
- ❌ **Templates divergentes sin includes**: todos los `tiers/*.j2` reutilizan `_project_metadata.j2` y `_reference_estimates.j2`.
- ❌ **Cambiar solo el tono**: cada tier cambia *estructura* + *vocabulario* + (en research) *pipeline*. Un Developer ve `components`, un Executive ve `headline_cost_range` y `go_no_go_recommendation`.

### Ejemplo de uso con `curl`

```bash
# Crear sesión
SID=$(curl -s -X POST http://127.0.0.1:8000/sessions | jq -r .session_id)

# Mismo brief, tres perfiles distintos
for TIER in developer pm executive; do
  curl -s -X POST "http://127.0.0.1:8000/sessions/$SID/estimate" \
    -H "X-Estimator-Tier: $TIER" \
    -H "X-Estimator-User: julio" \
    -d "transcript=Plataforma SaaS de gestión de gastos para PYMEs. Multi-tenant." \
    | jq '{tier, pipeline, structured_keys: (.structured | keys)}'
done
```

