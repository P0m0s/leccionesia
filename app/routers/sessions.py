"""
Router HTTP de sesiones conversacionales.

- ``POST /sessions``: crea una sesión nueva, devuelve ``session_id``.
- ``GET /sessions/{session_id}``: rehidrata historial + metadata.
- ``POST /sessions/{session_id}/estimate``: turno de chat con adjuntos
  opcionales (Camino B). Devuelve la estimación estructurada en JSON.
- ``POST /sessions/{session_id}/estimate/stream``: misma operación en streaming
  NDJSON (``token`` y ``final``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

ChatPromptVersion = Literal["v1", "v2"]

from app.schemas import (
    SessionCreateResponse,
    SessionEstimateResponse,
    SessionStateResponse,
)
from app.services.attachments import (
    MAX_ATTACHMENTS_PER_TURN,
    AttachmentImage,
    AttachmentText,
    encode_image_attachment,
    extract_attachment_text,
    is_image_attachment,
)
from app.services.session_service import (
    iter_session_turn_stream,
    run_session_turn,
)
from app.services.slash_commands import dispatch as dispatch_command
from app.services.slash_commands import is_command
from app.sessions import MAX_TURNS, session_store
from app.tiers import (
    PIPELINE_HANDLERS,
    CallerContext,
    get_caller_context,
    resolve_tier_config,
)

router = APIRouter(tags=["sessions"])


async def _read_and_extract(
    attachments: list[UploadFile] | None,
) -> tuple[list[AttachmentText], list[AttachmentImage], list[dict]]:
    """Lee uploads y clasifica:

    - **texts**: PDF/DOCX/TXT con texto extraído (Camino B).
    - **images**: PNG/JPEG/GIF/WEBP codificadas base64 (Camino A multimodal).
    - **failed**: cualquier adjunto que no se pudo procesar.
    """
    texts: list[AttachmentText] = []
    images: list[AttachmentImage] = []
    failed: list[dict] = []

    if not attachments:
        return texts, images, failed

    if len(attachments) > MAX_ATTACHMENTS_PER_TURN:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Demasiados adjuntos: {len(attachments)} "
                f"(máx. {MAX_ATTACHMENTS_PER_TURN} por turno)."
            ),
        )

    for upload in attachments:
        data = await upload.read()
        if not data:
            continue

        filename = upload.filename or "attachment"
        content_type = upload.content_type

        if is_image_attachment(filename, content_type):
            img = encode_image_attachment(
                filename=filename,
                content_type=content_type,
                data=data,
            )
            if img.error or not img.base64_data:
                failed.append({"filename": img.filename, "error": img.error or "Imagen vacía"})
            else:
                images.append(img)
            continue

        result = extract_attachment_text(
            filename=filename,
            content_type=content_type,
            data=data,
        )
        if result.error or not result.text:
            failed.append(
                {
                    "filename": result.filename,
                    "error": result.error or "Adjunto vacío tras extracción",
                },
            )
            continue
        texts.append(result)

    return texts, images, failed


@router.post("/sessions", response_model=SessionCreateResponse, status_code=201)
def create_session() -> SessionCreateResponse:
    session = session_store.create()
    return SessionCreateResponse(session_id=session.session_id)


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
def get_session(session_id: str) -> SessionStateResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesión desconocida")
    return SessionStateResponse(
        session_id=session.session_id,
        project_metadata=session.project_metadata.model_dump(),
        messages=[m.model_dump() for m in session.history.messages],
        max_turns=MAX_TURNS,
        metrics=session.metrics.model_dump(),
    )


@router.post(
    "/sessions/{session_id}/estimate",
    response_model=SessionEstimateResponse,
)
async def session_estimate(
    session_id: str,
    transcript: str = Form(default=""),
    attachments: list[UploadFile] | None = File(default=None),
    refine: bool = Query(
        default=False,
        description=(
            "Si True, hace una segunda pasada de auto-crítica antes de devolver "
            "la estimación. Duplica latencia y tokens. Solo aplica al flujo "
            "conversacional clásico (sin tier)."
        ),
    ),
    prompt_version: ChatPromptVersion = Query(
        default="v1",
        description="Versión del template conversacional (v1 estándar, v2 adversarial).",
    ),
    caller: CallerContext | None = Depends(get_caller_context),
) -> SessionEstimateResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesión desconocida")

    if is_command(transcript):
        cmd = dispatch_command(transcript, session)
        if cmd is not None and not cmd.needs_regenerate:
            return SessionEstimateResponse(
                session_id=session.session_id,
                text=cmd.text,
                structured=cmd.structured.model_dump(),
                structured_ok=cmd.structured_ok,
                project_metadata=session.project_metadata.model_dump(),
                metrics=session.metrics.model_dump(),
            )

        if cmd is not None and cmd.needs_regenerate:
            last_user = next(
                (m.content for m in reversed(session.history.messages) if m.role == "user"),
                None,
            )
            if last_user is None:
                raise HTTPException(
                    status_code=422,
                    detail="No hay turno de usuario previo para regenerar.",
                )
            session.history._messages.pop()
            transcript = last_user
            texts, images, failed = [], [], []
        else:
            texts, images, failed = [], [], []
    else:
        texts, images, failed = await _read_and_extract(attachments)

    if caller is not None:
        try:
            config = resolve_tier_config(caller.tier)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        handler = PIPELINE_HANDLERS[config["pipeline"]]
        try:
            tier_result = await handler(
                config=config,
                session=session,
                transcript=transcript,
                attachments=texts,
                images=images,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return SessionEstimateResponse(
            session_id=session.session_id,
            text=tier_result.summary_text,
            structured=tier_result.structured.model_dump(),
            structured_ok=True,
            refined=False,
            tier=caller.tier,
            pipeline=tier_result.pipeline,
            project_metadata=session.project_metadata.model_dump(),
            metrics=session.metrics.model_dump(),
            attachments_processed=[att.filename for att in texts] + [img.filename for img in images],
            attachments_failed=failed,
        )

    try:
        turn = run_session_turn(
            session=session,
            transcript=transcript,
            attachments=texts,
            images=images,
            prompt_version=prompt_version,
            refine=refine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SessionEstimateResponse(
        session_id=session.session_id,
        text=turn.text,
        structured=turn.structured.model_dump(),
        structured_ok=turn.structured_ok,
        refined=turn.refined,
        project_metadata=session.project_metadata.model_dump(),
        metrics=session.metrics.model_dump(),
        attachments_processed=[att.filename for att in texts] + [img.filename for img in images],
        attachments_failed=failed,
    )


@router.post("/sessions/{session_id}/estimate/stream")
async def session_estimate_stream(
    session_id: str,
    transcript: str = Form(default=""),
    attachments: list[UploadFile] | None = File(default=None),
    prompt_version: ChatPromptVersion = Query(default="v1"),
) -> StreamingResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesión desconocida")

    effective_transcript = transcript
    skip_attachments = False

    if is_command(transcript):
        cmd = dispatch_command(transcript, session)
        if cmd is not None and not cmd.needs_regenerate:

            def cmd_stream() -> Iterator[str]:
                yield json.dumps(
                    {
                        "type": "start",
                        "session_id": session.session_id,
                        "attachments_processed": [],
                        "attachments_failed": [],
                    },
                    ensure_ascii=False,
                ) + "\n"
                yield json.dumps(
                    {
                        "type": "final",
                        "session_id": session.session_id,
                        "text": cmd.text,
                        "structured": cmd.structured.model_dump(),
                        "structured_ok": cmd.structured_ok,
                        "project_metadata": session.project_metadata.model_dump(),
                        "metrics": session.metrics.model_dump(),
                    },
                    ensure_ascii=False,
                ) + "\n"

            return StreamingResponse(cmd_stream(), media_type="application/x-ndjson")

        if cmd is not None and cmd.needs_regenerate:
            last_user = next(
                (m.content for m in reversed(session.history.messages) if m.role == "user"),
                None,
            )
            if last_user is None:
                raise HTTPException(
                    status_code=422,
                    detail="No hay turno de usuario previo para regenerar.",
                )
            session.history._messages.pop()
            effective_transcript = last_user
            skip_attachments = True

    if skip_attachments:
        texts: list[AttachmentText] = []
        images: list[AttachmentImage] = []
        failed: list[dict] = []
    else:
        texts, images, failed = await _read_and_extract(attachments)
    processed_names = [att.filename for att in texts] + [img.filename for img in images]

    def event_stream() -> Iterator[str]:
        start_payload = {
            "type": "start",
            "session_id": session.session_id,
            "attachments_processed": processed_names,
            "attachments_failed": failed,
        }
        yield json.dumps(start_payload, ensure_ascii=False) + "\n"

        try:
            for event_type, payload in iter_session_turn_stream(
                session=session,
                transcript=effective_transcript,
                attachments=texts,
                images=images,
                prompt_version=prompt_version,
            ):
                line = {"type": event_type, **payload}
                yield json.dumps(line, ensure_ascii=False) + "\n"
        except ValueError as exc:
            yield json.dumps(
                {"type": "error", "error": str(exc)},
                ensure_ascii=False,
            ) + "\n"
        except Exception as exc:
            yield json.dumps(
                {"type": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )
