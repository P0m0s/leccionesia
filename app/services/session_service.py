"""
Orquestación de un turno de conversación dentro de una ``Session``.

Pasos por turno:

1. Cargar la sesión por ``session_id``.
2. Extraer texto de adjuntos (Camino B) y concatenarlo al transcript.
3. Añadir el mensaje del usuario al historial (ya con adjuntos embebidos).
4. Renderizar el system prompt vía Jinja2 inyectando ``project_metadata``.
5. Llamar al LLM con la lista [system, ...últimos N mensajes].
6. Añadir la respuesta al historial.
7. Actualizar ``project_metadata`` con la heurística regex sobre transcript + respuesta.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.config import settings
from app.prompts.loader import render_chat_refine_prompt
from app.services.attachments import (
    AttachmentImage,
    AttachmentText,
    build_augmented_transcript,
)
from app.services.cost import estimate_cost_usd
from app.services.llm_service import (
    ImageInput,
    generate_chat_messages,
    stream_chat_messages,
)
from app.services.metadata_extractor import update_metadata
from app.services.metadata_llm import (
    extract_metadata_via_llm,
    summarize_dropped_messages,
)
from app.sessions import ConversationMessage, ProjectMetadata, Session
from app.structured import StructuredEstimation, parse_structured_response

logger = structlog.get_logger(__name__)


@dataclass
class SessionTurnResult:
    text: str
    model: str
    provider: str
    augmented_transcript: str
    structured: StructuredEstimation
    structured_ok: bool
    refined: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


def _images_payload(images: list[AttachmentImage] | None) -> list[ImageInput]:
    if not images:
        return []
    return [(img.mime_type, img.base64_data) for img in images if img.base64_data]


def _append_user_capturing_overflow(
    session: Session,
    text: str,
    dropped: list[ConversationMessage],
) -> None:
    out = session.history.add_user_message(text)
    if out is not None:
        dropped.append(out)


def _append_assistant_capturing_overflow(
    session: Session,
    text: str,
    dropped: list[ConversationMessage],
) -> None:
    out = session.history.add_assistant_message(text)
    if out is not None:
        dropped.append(out)


def _maybe_summarize_dropped(
    session: Session,
    dropped: list[ConversationMessage],
) -> None:
    if not settings.enable_auto_summary or not dropped:
        return
    payload = [{"role": m.role, "content": m.content} for m in dropped]
    new_summary = summarize_dropped_messages(
        session.project_metadata.conversation_summary,
        payload,
    )
    if new_summary:
        session.project_metadata = session.project_metadata.merged_with(
            ProjectMetadata(conversation_summary=new_summary),
        )


def _maybe_enrich_metadata_via_llm(
    session: Session,
    user_text: str,
    assistant_text: str,
) -> None:
    if not settings.enable_llm_metadata:
        return
    patch = extract_metadata_via_llm(
        session.project_metadata,
        user_text,
        assistant_text,
    )
    if patch.model_dump(exclude_none=True, exclude_defaults=True):
        session.project_metadata = session.project_metadata.merged_with(patch)


def _accumulate_metrics(
    session: Session,
    metrics: dict[str, Any],
    *,
    n_attachments: int = 0,
    n_images: int = 0,
) -> None:
    sm = session.metrics
    sm.llm_calls += 1
    in_tok = metrics.get("input_tokens") or 0
    out_tok = metrics.get("output_tokens") or 0
    sm.input_tokens_total += int(in_tok)
    sm.output_tokens_total += int(out_tok)
    sm.elapsed_ms_total += float(metrics.get("elapsed_ms") or 0.0)
    sm.last_model = metrics.get("model")
    sm.last_provider = metrics.get("provider")
    sm.last_metrics = dict(metrics)

    cost = estimate_cost_usd(sm.last_model, in_tok, out_tok)
    if cost is not None:
        sm.estimated_cost_usd += cost

    sm.attachments_processed_total += n_attachments
    sm.images_processed_total += n_images


def run_session_turn(
    *,
    session: Session,
    transcript: str,
    attachments: list[AttachmentText],
    images: list[AttachmentImage] | None = None,
    prompt_version: str = "v1",
    refine: bool = False,
) -> SessionTurnResult:
    augmented = build_augmented_transcript(transcript, attachments)
    if not augmented and not images:
        msg = "El transcript (con o sin adjuntos) no puede estar vacío."
        raise ValueError(msg)
    if not augmented and images:
        augmented = "(El usuario ha enviado únicamente imagen(es); analízalas como contexto.)"

    dropped: list[ConversationMessage] = []
    _append_user_capturing_overflow(session, augmented, dropped)
    messages = session.history.to_messages_list(
        session.project_metadata,
        prompt_version=prompt_version,
    )

    logger.info(
        "session_turn_llm_call",
        session_id=session.session_id,
        n_messages=len(messages),
        attachments=[att.filename for att in attachments],
        n_images=len(images or []),
        refine=refine,
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
    structured, structured_ok = parse_structured_response(raw_text)
    visible_text = structured.summary_markdown or raw_text
    refined = False

    if refine:
        critique_prompt = render_chat_refine_prompt(version=prompt_version)
        critique_messages = messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": critique_prompt},
        ]
        logger.info("session_turn_refine_call", session_id=session.session_id)
        refine_metrics: dict[str, Any] = {}
        raw_text_2, _model2, _prov2 = generate_chat_messages(
            critique_messages,
            metrics_out=refine_metrics,
        )
        _accumulate_metrics(session, refine_metrics)
        structured_2, ok_2 = parse_structured_response(raw_text_2)
        if ok_2 or structured_2.summary_markdown:
            structured = structured_2
            structured_ok = ok_2
            visible_text = structured.summary_markdown or raw_text_2
            refined = True

    _append_assistant_capturing_overflow(session, visible_text, dropped)

    session.project_metadata = update_metadata(
        session.project_metadata,
        augmented,
        visible_text,
    )
    session.metrics.turns_count += 1
    _maybe_summarize_dropped(session, dropped)
    _maybe_enrich_metadata_via_llm(session, augmented, visible_text)

    logger.info(
        "session_turn_parsed",
        session_id=session.session_id,
        structured_ok=structured_ok,
        n_line_items=len(structured.line_items),
        n_phases=len(structured.phases),
        confidence=structured.confidence,
        refined=refined,
    )

    return SessionTurnResult(
        text=visible_text,
        model=model,
        provider=provider,
        augmented_transcript=augmented,
        structured=structured,
        structured_ok=structured_ok,
        refined=refined,
    )


def iter_session_turn_stream(
    *,
    session: Session,
    transcript: str,
    attachments: list[AttachmentText],
    images: list[AttachmentImage] | None = None,
    prompt_version: str = "v1",
) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Streaming del turno. Yield de tuplas ``(event_type, payload)`` para que el
    router las serialice como NDJSON.

    Eventos emitidos:

    - ``token``: ``{"delta": "<chunk>"}`` por cada chunk del LLM.
    - ``final``: ``{"text": ..., "structured": {...}, "structured_ok": bool,
       "project_metadata": {...}, "metrics": {...}}``.
    """
    augmented = build_augmented_transcript(transcript, attachments)
    if not augmented and not images:
        msg = "El transcript (con o sin adjuntos) no puede estar vacío."
        raise ValueError(msg)
    if not augmented and images:
        augmented = "(El usuario ha enviado únicamente imagen(es); analízalas como contexto.)"

    dropped: list[ConversationMessage] = []
    _append_user_capturing_overflow(session, augmented, dropped)
    messages = session.history.to_messages_list(
        session.project_metadata,
        prompt_version=prompt_version,
    )

    logger.info(
        "session_turn_stream_llm_call",
        session_id=session.session_id,
        n_messages=len(messages),
        attachments=[att.filename for att in attachments],
        n_images=len(images or []),
    )

    metrics: dict[str, Any] = {}
    chunks: list[str] = []
    for chunk in stream_chat_messages(
        messages,
        metrics_out=metrics,
        images=_images_payload(images),
    ):
        if chunk:
            chunks.append(chunk)
            yield "token", {"delta": chunk}

    _accumulate_metrics(
        session,
        metrics,
        n_attachments=len(attachments),
        n_images=len(images or []),
    )

    raw_text = "".join(chunks)
    structured, structured_ok = parse_structured_response(raw_text)
    visible_text = structured.summary_markdown or raw_text

    _append_assistant_capturing_overflow(session, visible_text, dropped)
    session.project_metadata = update_metadata(
        session.project_metadata,
        augmented,
        visible_text,
    )
    session.metrics.turns_count += 1
    _maybe_summarize_dropped(session, dropped)
    _maybe_enrich_metadata_via_llm(session, augmented, visible_text)

    logger.info(
        "session_turn_stream_parsed",
        session_id=session.session_id,
        structured_ok=structured_ok,
        confidence=structured.confidence,
    )

    yield "final", {
        "session_id": session.session_id,
        "text": visible_text,
        "structured": structured.model_dump(),
        "structured_ok": structured_ok,
        "project_metadata": session.project_metadata.model_dump(),
        "metrics": session.metrics.model_dump(),
    }
