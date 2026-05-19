"""Tests de adjuntos imagen (Camino A multimodal)."""

from __future__ import annotations

import base64

import pytest
from httpx import AsyncClient

from app.services.attachments import (
    encode_image_attachment,
    is_image_attachment,
)
from app.services.llm_service import (
    _inject_images_anthropic,
    _inject_images_openai,
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9ZqQEoUAAAAASUVORK5CYII=",
)


def test_detecta_imagen_por_extension_y_mime() -> None:
    assert is_image_attachment("diagrama.png", None)
    assert is_image_attachment("foto.jpg", "image/jpeg")
    assert is_image_attachment("a.webp", None)
    assert not is_image_attachment("doc.pdf", "application/pdf")
    assert not is_image_attachment("notas.txt", "text/plain")


def test_encode_image_devuelve_base64_y_mime() -> None:
    img = encode_image_attachment(
        filename="pix.png",
        content_type="image/png",
        data=PNG_1X1,
    )
    assert img.error is None
    assert img.mime_type == "image/png"
    assert img.base64_data
    assert base64.b64decode(img.base64_data) == PNG_1X1


def test_inject_images_openai_convierte_ultimo_user_a_multimodal() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Hola"},
    ]
    images = [("image/png", "BASE64DATA")]
    out = _inject_images_openai(messages, images)
    user_msg = out[-1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0] == {"type": "text", "text": "Hola"}
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_inject_images_anthropic_convierte_ultimo_user_a_blocks() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Hola"},
    ]
    images = [("image/png", "BASE64DATA")]
    out = _inject_images_anthropic(messages, images)
    user_msg = out[-1]
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "image"
    assert user_msg["content"][0]["source"]["media_type"] == "image/png"
    assert user_msg["content"][-1]["type"] == "text"


def test_inject_images_vacio_no_modifica() -> None:
    messages = [{"role": "user", "content": "hola"}]
    out = _inject_images_openai(messages, [])
    assert out == messages


@pytest.mark.asyncio
async def test_endpoint_acepta_imagen_y_la_pasa_al_llm(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "¿Qué muestra este diagrama?"},
        files=[("attachments", ("diagrama.png", PNG_1X1, "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["attachments_processed"] == ["diagrama.png"]
    images_passed = fake_llm["images_per_call"][-1]
    assert len(images_passed) == 1
    mime, b64 = images_passed[0]
    assert mime == "image/png"
    assert base64.b64decode(b64) == PNG_1X1


@pytest.mark.asyncio
async def test_endpoint_acepta_solo_imagen_sin_transcript(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    """El servicio sustituye el transcript vacío por una nota explicativa."""
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": ""},
        files=[("attachments", ("diagrama.png", PNG_1X1, "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    assert fake_llm["images_per_call"][-1]
