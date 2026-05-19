"""Tests del extractor LLM complementario y del resumen automático."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.config import settings
from app.sessions import ProjectMetadata, session_store


@pytest.fixture
def enable_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_auto_summary", True)
    monkeypatch.setattr(settings, "enable_llm_metadata", False)


@pytest.fixture
def enable_metadata_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_auto_summary", False)
    monkeypatch.setattr(settings, "enable_llm_metadata", True)


@pytest.mark.asyncio
async def test_auto_summary_se_invoca_cuando_overflow(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm: dict,
    enable_summary,
) -> None:
    """Tras MAX_TURNS, las nuevas llamadas deberían disparar el summarizer."""
    summarize_calls: list[list[dict]] = []

    def fake_summarize(previous, dropped_messages):
        summarize_calls.append(list(dropped_messages))
        return "Resumen artificial actualizado."

    monkeypatch.setattr(
        "app.services.session_service.summarize_dropped_messages",
        fake_summarize,
    )

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    for i in range(3):
        await async_client.post(
            f"/sessions/{sid}/estimate",
            data={"transcript": f"Turno {i}."},
        )
    assert summarize_calls == []

    for i in range(3, 8):
        await async_client.post(
            f"/sessions/{sid}/estimate",
            data={"transcript": f"Turno {i}."},
        )

    assert summarize_calls
    session = session_store.get(sid)
    assert session is not None
    assert session.project_metadata.conversation_summary == "Resumen artificial actualizado."
    first_dropped = summarize_calls[0]
    assert any("Turno 0." in m["content"] for m in first_dropped)


@pytest.mark.asyncio
async def test_metadata_llm_patches_se_fusionan(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm: dict,
    enable_metadata_llm,
) -> None:
    def fake_extract(current, user_msg, assistant_msg):
        return ProjectMetadata(
            project_name="LLM-detected",
            mentioned_technologies=["Astro", "Cloudflare"],
            agreed_scope="alcance refinado por LLM",
        )

    monkeypatch.setattr(
        "app.services.session_service.extract_metadata_via_llm",
        fake_extract,
    )

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]
    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Hola"},
    )
    assert resp.status_code == 200
    metadata = resp.json()["project_metadata"]
    assert metadata["project_name"] == "LLM-detected"
    assert "Astro" in metadata["mentioned_technologies"]
    assert "Cloudflare" in metadata["mentioned_technologies"]
    assert metadata["agreed_scope"] == "alcance refinado por LLM"


def test_conversation_summary_aparece_en_system_prompt() -> None:
    from app.prompts.loader import render_chat_system_prompt

    pm = ProjectMetadata(conversation_summary="Acuerdo previo: MVP en 6 semanas.")
    rendered = render_chat_system_prompt(pm)
    assert "conversation_summary" in rendered
    assert "MVP en 6 semanas" in rendered


def test_json_robusto_a_basura_del_extractor_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si el LLM devuelve basura, el extractor devuelve metadata vacía sin romper."""
    from app.services.metadata_llm import extract_metadata_via_llm

    def fake_generate(messages, **_kwargs):
        return "esto no es json", "fake", "fake"

    monkeypatch.setattr(
        "app.services.metadata_llm.generate_chat_messages",
        fake_generate,
    )

    patch = extract_metadata_via_llm(
        ProjectMetadata(project_name="X"),
        user_message="u",
        assistant_message="a",
    )
    assert patch.model_dump(exclude_none=True, exclude_defaults=True) == {}
