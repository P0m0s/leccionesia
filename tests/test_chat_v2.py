"""Tests del prompt conversacional v2 (filosofía adversarial)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.prompts.loader import (
    render_chat_refine_prompt,
    render_chat_system_prompt,
)
from app.sessions import ProjectMetadata


def test_chat_v2_system_prompt_es_adversarial() -> None:
    rendered = render_chat_system_prompt(ProjectMetadata(), version="v2")
    assert "adversarial" in rendered.lower()
    assert "Preguntas críticas" in rendered or "preguntas críticas" in rendered.lower()
    assert "<project_metadata>" in rendered
    assert "summary_markdown" in rendered


def test_chat_v2_refine_prompt() -> None:
    rendered = render_chat_refine_prompt(version="v2")
    assert "adversarial" in rendered.lower()


def test_chat_v2_examples_se_resuelven_via_fallback_a_v1() -> None:
    """v2 no tiene su propio examples/; debe caer en los de v1."""
    pm = ProjectMetadata(project_type="data_pipeline")
    rendered = render_chat_system_prompt(pm, version="v2")
    assert "data_pipeline" in rendered


def test_version_desconocida_lanza_valueerror() -> None:
    with pytest.raises(ValueError, match="Versión de prompt de chat desconocida"):
        render_chat_system_prompt(ProjectMetadata(), version="v99")


@pytest.mark.asyncio
async def test_endpoint_acepta_prompt_version_v2(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]
    resp = await async_client.post(
        f"/sessions/{sid}/estimate?prompt_version=v2",
        data={"transcript": "Quiero estimar un MVP."},
    )
    assert resp.status_code == 200, resp.text

    system_msg = fake_llm["calls"][0][0]
    assert system_msg["role"] == "system"
    assert "adversarial" in system_msg["content"].lower()
