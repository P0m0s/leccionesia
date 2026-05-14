# Estimador CAG (FastAPI)

Servicio REST que recibe la **transcripción** de una reunión y devuelve una **estimación** generada por un LLM, usando arquitectura **CAG** (ejemplos estáticos en el prompt).

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Cuenta OpenAI o Anthropic y API key

## Configuración

```bash
cp .env.example .env
```

Edita `.env` con tus claves y, si aplica, `LLM_PROVIDER`, `OPENAI_MODEL` y `ANTHROPIC_MODEL`.

## Instalación y arranque

Desde la raíz del proyecto, con **uv** (recomendado en el curso):

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Sin `uv`, con **venv + pip**:

```bash
py -m venv .venv
.\.venv\Scripts\pip install -e .
.\.venv\Scripts\uvicorn app.main:app --reload
```

- Documentación interactiva: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Salud: `GET /health`

## Probar el endpoint

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/estimate" ^
  -H "Content-Type: application/json" ^
  -d "{\"transcription\": \"El cliente quiere un portal de facturación electrónica con integración SUNAT y reportes.\"}"
```

En PowerShell puedes usar `Invoke-RestMethod` en lugar de `curl` si prefieres.

## Estructura

- `app/main.py` — aplicación FastAPI y `GET /health`
- `app/config.py` — variables de entorno con `pydantic-settings`
- `app/routers/estimations.py` — `POST /api/v1/estimate`
- `app/services/llm_service.py` — prompt + llamada OpenAI/Anthropic
- `app/context/examples.py` — ejemplos few-shot

## Checklist

- [ ] Servidor arranca sin errores
- [ ] API keys en `.env`
- [ ] `GET /health` → 200
- [ ] `POST /api/v1/estimate` devuelve estimación
- [ ] Swagger en `/docs`
- [ ] `.env` no versionado (está en `.gitignore`)
