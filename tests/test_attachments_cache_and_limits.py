"""Tests del caché por SHA-256 y de los límites de adjuntos."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.services.attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS_PER_TURN,
    cache_size,
    extract_attachment_text,
    reset_cache,
)


def test_cache_evita_re_extraccion_del_mismo_binario() -> None:
    reset_cache()
    data = b"Hola, este es un texto plano de prueba."
    r1 = extract_attachment_text(
        filename="nota.txt",
        content_type="text/plain",
        data=data,
    )
    assert r1.text and r1.error is None
    assert cache_size() == 1

    r2 = extract_attachment_text(
        filename="otra_copia.txt",
        content_type="text/plain",
        data=data,
    )
    assert r2.text == r1.text
    assert cache_size() == 1


def test_rechaza_archivo_por_encima_del_limite() -> None:
    reset_cache()
    oversized = b"x" * (MAX_ATTACHMENT_BYTES + 1)
    result = extract_attachment_text(
        filename="enorme.txt",
        content_type="text/plain",
        data=oversized,
    )
    assert result.error is not None
    assert "demasiado grande" in result.error.lower()
    assert result.text == ""


@pytest.mark.asyncio
async def test_rechaza_demasiados_adjuntos_por_turno(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    files = [
        ("attachments", (f"f{i}.txt", b"contenido", "text/plain"))
        for i in range(MAX_ATTACHMENTS_PER_TURN + 1)
    ]
    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Estimar algo."},
        files=files,
    )
    assert resp.status_code == 413
    assert "Demasiados adjuntos" in resp.json()["detail"]
