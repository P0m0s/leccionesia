"""
Capa LLM **complementaria** sobre la heurística regex:

- ``extract_metadata_via_llm``: tras un turno, hace una segunda llamada ligera
  al LLM con un prompt-extractor que devuelve JSON con la metadata estructurada
  (nombre, tipo, equipo, tecnologías, alcance, resumen). Se mezcla con la
  metadata regex usando ``ProjectMetadata.merged_with``.
- ``summarize_dropped_messages``: cuando la ventana deslizante deja caer un
  mensaje antiguo, lo resume y acumula en ``conversation_summary`` para no
  perder contexto histórico.

Ambas funciones son tolerantes a fallos: si el LLM no devuelve JSON válido o
falla la red, devuelven valores neutros (no rompen el turno).
"""

from __future__ import annotations

import json

import structlog

from app.services.llm_service import generate_chat_messages
from app.sessions import ProjectMetadata
from app.structured import _extract_json_payload

logger = structlog.get_logger(__name__)


_METADATA_EXTRACTOR_SYSTEM = """\
Eres un extractor estructurado. Recibirás:
- el `project_metadata` actual,
- el último turno (transcripción del usuario + respuesta del asistente).

Tu trabajo es devolver UN único JSON con campos NUEVOS o REFINADOS para
`project_metadata`. NO repitas información que ya esté en el `project_metadata`
actual y sea idéntica. Si no hay novedades, devuelve un objeto vacío `{}`.

Schema (todos los campos son opcionales):
{
  "project_name": "string | null",
  "project_type": "mobile_app | web_saas | internal_tool | data_pipeline | null",
  "assumed_team_size": integer | null,
  "mentioned_technologies": ["string"],
  "agreed_scope": "string corto | null"
}

Reglas:
- NO inventes datos. Sólo extrae lo que esté presente.
- `mentioned_technologies` solo debe contener TECNOLOGÍAS reales (lenguajes,
  frameworks, servicios). Una sola lista, sin descripciones.
- Responde SOLO con JSON. Sin texto fuera, sin code fences.
"""


_SUMMARIZER_SYSTEM = """\
Eres un compresor de contexto conversacional. Recibirás:
- un resumen acumulado previo (puede estar vacío),
- los mensajes (turnos) que están a punto de salir de la ventana deslizante.

Tu trabajo es devolver un nuevo `conversation_summary` actualizado, en español,
máximo 4 frases, que preserve:
- decisiones acordadas,
- supuestos asumidos,
- temas pendientes.

NO incluyas marcadores temporales ni listas. Texto plano.
"""


def extract_metadata_via_llm(
    current_metadata: ProjectMetadata,
    user_message: str,
    assistant_message: str,
) -> ProjectMetadata:
    """Devuelve un *parche* de metadata extraído por el LLM. Tolerante a fallos."""
    user_payload = (
        "project_metadata actual:\n"
        f"{current_metadata.model_dump_json(indent=2)}\n\n"
        "Último turno:\n"
        f"USER: {user_message}\n"
        f"ASSISTANT: {assistant_message}"
    )
    try:
        raw, _model, _provider = generate_chat_messages(
            [
                {"role": "system", "content": _METADATA_EXTRACTOR_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
        )
    except Exception as exc:
        logger.warning("metadata_llm_call_failed", error=str(exc))
        return ProjectMetadata()

    payload = _extract_json_payload(raw)
    if payload is None:
        return ProjectMetadata()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ProjectMetadata()

    try:
        return ProjectMetadata.model_validate(data)
    except Exception as exc:
        logger.warning("metadata_llm_validation_failed", error=str(exc), data=data)
        return ProjectMetadata()


def summarize_dropped_messages(
    previous_summary: str | None,
    dropped_messages: list[dict[str, str]],
) -> str | None:
    """Genera un resumen actualizado con los mensajes que dejan la ventana.

    Si el LLM falla, devuelve el resumen previo (sin actualizar) para no
    bloquear el turno.
    """
    if not dropped_messages:
        return previous_summary

    blocks = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in dropped_messages
    )
    user_payload = (
        "Resumen previo:\n"
        f"{previous_summary or '(vacío)'}\n\n"
        "Mensajes que salen de la ventana:\n"
        f"{blocks}"
    )
    try:
        raw, _model, _prov = generate_chat_messages(
            [
                {"role": "system", "content": _SUMMARIZER_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
        )
    except Exception as exc:
        logger.warning("summarize_dropped_messages_failed", error=str(exc))
        return previous_summary

    return raw.strip() or previous_summary
