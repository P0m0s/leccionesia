"""Test 2: un adjunto (PDF) hace que la estimación cambie respecto a sin adjunto."""

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)


def _build_pdf_with_text(text: str) -> bytes:
    """Construye un PDF de 1 página con el texto dado, sin dependencias extra."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    font_dict = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        },
    )
    font_ref = writer._add_object(font_dict)
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_ref},
            ),
        },
    )
    page[NameObject("/Resources")] = resources

    safe = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    content_stream = (
        "BT\n"
        "/F1 12 Tf\n"
        "72 720 Td\n"
        f"({safe}) Tj\n"
        "ET\n"
    ).encode("latin-1")
    stream_obj = DecodedStreamObject()
    stream_obj.set_data(content_stream)
    stream_ref = writer._add_object(stream_obj)
    page[NameObject("/Contents")] = stream_ref
    page[NameObject("/MediaBox")] = ArrayObject(
        [NumberObject(0), NumberObject(0), NumberObject(612), NumberObject(792)],
    )

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_helper_round_trip() -> None:
    """Sanity: el PDF generado contiene el texto esperado al extraerlo."""
    pdf_bytes = _build_pdf_with_text("Hola Aurora")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 1
    assert "Hola" in (reader.pages[0].extract_text() or "")


@pytest.mark.asyncio
async def test_adjunto_pdf_cambia_la_estimacion(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    pdf_bytes = _build_pdf_with_text(
        "Documento anexo: integracion con Stripe y notificaciones SMS via Twilio."
        " Equipo de 5 personas."
    )

    create_a = await async_client.post("/sessions")
    sid_a = create_a.json()["session_id"]
    resp_no_attach = await async_client.post(
        f"/sessions/{sid_a}/estimate",
        data={"transcript": "Necesito estimar un MVP de marketplace."},
    )
    assert resp_no_attach.status_code == 200, resp_no_attach.text
    body_no_attach = resp_no_attach.json()
    text_no_attach = body_no_attach["text"]

    create_b = await async_client.post("/sessions")
    sid_b = create_b.json()["session_id"]
    resp_with_attach = await async_client.post(
        f"/sessions/{sid_b}/estimate",
        data={"transcript": "Necesito estimar un MVP de marketplace."},
        files=[("attachments", ("anexo.pdf", pdf_bytes, "application/pdf"))],
    )
    assert resp_with_attach.status_code == 200, resp_with_attach.text
    body_with_attach = resp_with_attach.json()
    text_with_attach = body_with_attach["text"]

    assert body_with_attach["attachments_processed"] == ["anexo.pdf"]
    assert body_with_attach["attachments_failed"] == []

    assert text_no_attach != text_with_attach

    last_call_no_attach = fake_llm["calls"][0]
    last_call_with_attach = fake_llm["calls"][1]
    last_user_no = next(m for m in reversed(last_call_no_attach) if m["role"] == "user")
    last_user_yes = next(m for m in reversed(last_call_with_attach) if m["role"] == "user")
    assert "--- attachment:" not in last_user_no["content"]
    assert "--- attachment: anexo.pdf ---" in last_user_yes["content"]
    assert "Stripe" in last_user_yes["content"] or "Twilio" in last_user_yes["content"]

    assert "Stripe" in body_with_attach["project_metadata"]["mentioned_technologies"]
    assert "Twilio" in body_with_attach["project_metadata"]["mentioned_technologies"]
    assert body_with_attach["project_metadata"]["assumed_team_size"] == 5
