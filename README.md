# Estimador CAG (FastAPI + Streamlit)

API REST que genera **estimaciones de software** con un LLM usando **CAG** (contexto estático): las instrucciones y ejemplos few-shot viven en plantillas **Jinja2** versionadas (`v1`, `v2`), no en código Python disperso. El servicio admite **OpenAI** o **Anthropic** según configuración, **caché exact-match** sobre la petición serializada y, opcionalmente, **respuesta en streaming**.

La interfaz **Streamlit** envía el mismo contrato JSON que la API (formulario tipado + historial + streaming HTTP), de modo que no se duplica la lógica de prompts ni de proveedores en el cliente.

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

### Caché y versiones de prompt

- **Caché exact-match:** si dos peticiones tienen el mismo JSON canónico (mismos campos y valores) y la misma `prompt_version`, la segunda puede devolver el texto cacheado sin llamar otra vez al LLM.
- **Versiones:** `v1` y `v2` corresponden a carpetas bajo `app/prompts/estimation/`. Puedes añadir más versiones ampliando el loader y las plantillas.

## Tests

Sin llamadas a APIs externas (solo render Jinja):

```bash
uv run pytest tests/prompts/test_estimation_v1.py -v
```

## Estructura del proyecto (resumen)

| Ruta | Rol |
|------|-----|
| `app/main.py` | FastAPI, configuración mínima de `structlog`. |
| `app/config.py` | Ajustes desde entorno (`pydantic-settings`). |
| `app/schemas.py` | Modelos Pydantic de entrada y salida. |
| `app/routers/estimations.py` | `POST /estimate` y `POST /estimate/stream`. |
| `app/prompts/loader.py` | Render de `system.j2` + `user.j2` por versión. |
| `app/prompts/estimation/v1/` … `v2/` | Plantillas Jinja2 (CAG versionado). |
| `app/services/llm_service.py` | Wrapper multi-proveedor (mensajes system + user). |
| `app/services/estimate_service.py` | Orquestación: render, caché, llamada al LLM. |
| `streamlit_app.py` | Formulario, streaming HTTP e historial. |
| `tests/prompts/` | Tests del render de plantillas. |

El archivo `app/context/examples.py` es legado del primer ejercicio; los ejemplos few-shot activos están en los `.j2` dentro de `app/prompts/`.

## Checklist rápido

- [ ] `.env` creado a partir de `.env.example` y clave del proveedor elegido rellenada
- [ ] `GET /health` responde 200
- [ ] `POST /estimate` con un JSON válido devuelve `text` y `prompt_version`
- [ ] `/docs` abre y el esquema coincide con `EstimationRequest`
- [ ] Streamlit puede alcanzar la API (`ESTIMATOR_API_BASE` si no usas el puerto por defecto)
- [ ] `.env` no se sube al repositorio (suele ignorarse con `.gitignore`)
