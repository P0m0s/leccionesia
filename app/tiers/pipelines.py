"""Orquestadores (`pipelines`) seleccionables por tier.

Cada pipeline recibe:

- `config`: el dict de `TIER_CONFIG[tier]`.
- `session`: la `Session` en curso (memoria conversacional + métricas).
- `transcript`: turno del usuario.
- `attachments`: lista de `AttachmentText` ya extraídos (Camino B).
- `images`: lista opcional de `AttachmentImage` (Camino A multimodal).

Y devuelve un `TierTurnResult` con:

- `structured`: instancia del schema Pydantic del tier (ya validado).
- `summary_text`: representación textual breve para mostrar en el historial
  conversacional (no es la salida principal, que es el JSON).
- `model`, `provider`: usados en el LLM call.
- `pipeline`: nombre de la pipeline ejecutada.

El sistema mantiene **memoria conversacional** también en modo tier: cada
turno se añade al historial igual que en el flujo conversacional clásico, de
forma que un usuario puede alternar de tier dentro de la misma sesión.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ValidationError

from app.prompts.loader import _environment, _format_project_metadata
from app.services.attachments import build_augmented_transcript
from app.services.llm_service import generate_chat_messages
from app.services.metadata_extractor import update_metadata
from app.services.session_service import (
    _accumulate_metrics,
    _append_assistant_capturing_overflow,
    _append_user_capturing_overflow,
    _images_payload,
    _maybe_enrich_metadata_via_llm,
    _maybe_summarize_dropped,
)
from app.sessions import ConversationMessage
from app.structured import _try_parse_json

if TYPE_CHECKING:
    from app.services.attachments import AttachmentImage, AttachmentText
    from app.sessions import Session

logger = structlog.get_logger(__name__)


@dataclass
class TierTurnResult:
    """Resultado de un turno tier-aware."""

    structured: BaseModel
    summary_text: str
    model: str
    provider: str
    pipeline: str


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _render_tier_system_prompt(template_path: str, session: Session) -> str:
    """Renderiza el template Jinja2 del tier inyectando los parciales comunes."""
    env = _environment()
    template = env.get_template(template_path)
    return template.render(
        project_metadata=_format_project_metadata(session.project_metadata),
    )


def _validate_with_schema(
    raw_text: str,
    schema: type[BaseModel],
) -> tuple[BaseModel | None, str | None]:
    """Intenta parsear ``raw_text`` y validarlo con ``schema``.

    Devuelve ``(instancia, error)``. Si todo OK, ``error`` es ``None``.
    """
    payload = _extract_top_level_json(raw_text)
    if payload is None:
        return None, "El modelo no devolvió un objeto JSON parseable."

    data = _try_parse_json(payload)
    if data is None:
        return None, "JSON inválido en la respuesta del modelo."

    try:
        return schema.model_validate(data), None
    except ValidationError as e:
        return None, f"Schema {schema.__name__} no validado: {e.error_count()} errores."


def _extract_top_level_json(text: str) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        return text[first : last + 1]
    return None


def _summary_text_for_history(tier: str, structured: BaseModel) -> str:
    """Genera una representación textual breve para almacenar en el historial.

    Cada tier devuelve estructuras distintas, así que el "qué mostrar en el
    chat" se decide aquí, no en el frontend.
    """
    data = structured.model_dump()

    if tier == "developer":
        n = len(data.get("components", []))
        rng = data.get("total_hours_range") or {}
        return (
            f"**Estimación developer:** {n} componentes · "
            f"{rng.get('min', '?')}-{rng.get('max', '?')} h."
        )

    if tier == "pm":
        n_phases = len(data.get("phases", []))
        n_ms = len(data.get("milestones", []))
        rng = data.get("duration_weeks_range") or {}
        return (
            f"**Estimación PM:** {n_phases} fases · {n_ms} hitos · "
            f"{rng.get('min', '?')}-{rng.get('max', '?')} semanas."
        )

    if tier == "executive":
        cost = data.get("headline_cost_range") or {}
        dur = data.get("headline_duration_range") or {}
        rec = data.get("go_no_go_recommendation", "?")
        return (
            f"**Estimación ejecutiva:** {cost.get('min', '?')}-"
            f"{cost.get('max', '?')} € · "
            f"{dur.get('min', '?')}-{dur.get('max', '?')} semanas · "
            f"recomendación: **{rec}**."
        )

    if tier == "research":
        title = data.get("title") or "Informe de investigación"
        n_sections = len(data.get("sections", []))
        return f"**{title}** — informe de investigación ({n_sections} secciones)."

    return "Estimación generada."


def _build_messages_for_turn(
    session: Session,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Construye [system, ...history] reusando el historial del propio Session."""
    msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    msgs.extend(
        {"role": m.role, "content": m.content}
        for m in session.history.messages
    )
    return msgs


# ---------------------------------------------------------------------------
# Pipeline: single_call
# ---------------------------------------------------------------------------


async def run_single_call(
    *,
    config: dict[str, Any],
    session: Session,
    transcript: str,
    attachments: list[AttachmentText],
    images: list[AttachmentImage] | None = None,
) -> TierTurnResult:
    """Pipeline más simple: 1 llamada al LLM y validación de schema."""
    augmented = build_augmented_transcript(transcript, attachments)
    if not augmented and not images:
        msg = "El transcript (con o sin adjuntos) no puede estar vacío."
        raise ValueError(msg)
    if not augmented and images:
        augmented = "(El usuario ha enviado únicamente imagen(es); analízalas como contexto.)"

    dropped: list[ConversationMessage] = []
    _append_user_capturing_overflow(session, augmented, dropped)

    system_prompt = _render_tier_system_prompt(config["template"], session)
    messages = _build_messages_for_turn(session, system_prompt)

    logger.info(
        "tier_pipeline_call",
        pipeline="single_call",
        tier_template=config["template"],
        schema=config["schema"].__name__,
        n_messages=len(messages),
        n_attachments=len(attachments),
        n_images=len(images or []),
    )

    metrics: dict[str, Any] = {}
    raw_text, model, provider = generate_chat_messages(
        messages,
        images=_images_payload(images),
        metrics_out=metrics,
    )
    _accumulate_metrics(
        session,
        metrics,
        n_attachments=len(attachments),
        n_images=len(images or []),
    )

    structured, error = _validate_with_schema(raw_text, config["schema"])
    if structured is None:
        logger.warning(
            "tier_pipeline_validation_failed",
            error=error,
            raw_preview=raw_text[:300],
        )
        msg = (
            f"El modelo no devolvió un JSON conforme al schema "
            f"{config['schema'].__name__}: {error}"
        )
        raise ValueError(msg)

    summary_text = _summary_text_for_history(_tier_from_config(config), structured)
    _append_assistant_capturing_overflow(session, summary_text, dropped)

    session.project_metadata = update_metadata(
        session.project_metadata,
        augmented,
        summary_text,
    )
    session.metrics.turns_count += 1
    _maybe_summarize_dropped(session, dropped)
    _maybe_enrich_metadata_via_llm(session, augmented, summary_text)

    return TierTurnResult(
        structured=structured,
        summary_text=summary_text,
        model=model,
        provider=provider,
        pipeline="single_call",
    )


# ---------------------------------------------------------------------------
# Pipeline: deep_research
# ---------------------------------------------------------------------------


async def run_deep_research(
    *,
    config: dict[str, Any],
    session: Session,
    transcript: str,
    attachments: list[AttachmentText],
    images: list[AttachmentImage] | None = None,
) -> TierTurnResult:
    """Pipeline para informes profundos.

    En producción usaría ``o3-deep-research`` con `web_search` y ejecución en
    background. Mientras no haya acceso a esa API, hacemos una llamada
    síncrona al modelo configurado por defecto (el provider activo) usando
    el template `research.j2`, que ya pide un informe extenso con índice,
    secciones y citas.

    El resto del contrato (validación del schema, memoria conversacional,
    métricas) es idéntico a ``single_call``. Solo cambian: el template y el
    schema, ambos resueltos por `TIER_CONFIG`.
    """
    logger.info(
        "tier_pipeline_call",
        pipeline="deep_research",
        background_intended=config.get("background"),
        estimated_latency_seconds=config.get("estimated_latency_seconds"),
        note="usando provider por defecto hasta integrar API o3-deep-research",
    )
    result = await run_single_call(
        config=config,
        session=session,
        transcript=transcript,
        attachments=attachments,
        images=images,
    )
    return TierTurnResult(
        structured=result.structured,
        summary_text=result.summary_text,
        model=result.model,
        provider=result.provider,
        pipeline="deep_research",
    )


def _tier_from_config(config: dict[str, Any]) -> str:
    """Recupera el nombre del tier a partir de la config (para logs y summary)."""
    schema_name = config["schema"].__name__
    mapping = {
        "DeveloperEstimate": "developer",
        "PmEstimate": "pm",
        "ExecutiveEstimate": "executive",
        "ResearchEstimate": "research",
    }
    return mapping.get(schema_name, "unknown")


# `json` importado por si añadimos un futuro fallback que recree JSON.
_ = json


__all__ = [
    "TierTurnResult",
    "run_deep_research",
    "run_single_call",
]
