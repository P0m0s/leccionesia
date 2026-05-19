"""Tests del endpoint de streaming NDJSON y GET /sessions/{id}."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient


def _parse_ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_stream_emite_start_tokens_y_final(
    async_client: AsyncClient,
    fake_llm_stream: dict,
) -> None:
    fake_llm_stream["text"] = (
        '{"summary_markdown": "## Resultado streaming", '
        '"line_items": [{"name": "Auth", "area": "backend", '
        '"t_shirt": "S", "hours_min": 16, "hours_max": 40}], '
        '"phases": [], "assumptions": [], "risks": [], '
        '"confidence": 7, "next_step": "Decidir pasarela"}'
    )
    fake_llm_stream["chunk_size"] = 16

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate/stream",
        data={"transcript": "Quiero estimar una app SaaS con auth."},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    events = _parse_ndjson(resp.text)
    assert events[0]["type"] == "start"
    assert events[0]["session_id"] == sid

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) >= 2
    assembled = "".join(e["delta"] for e in token_events)
    assert "Resultado streaming" in assembled

    final = next(e for e in events if e["type"] == "final")
    assert final["text"] == "## Resultado streaming"
    assert final["structured_ok"] is True
    assert final["structured"]["confidence"] == 7
    assert final["structured"]["total_hours_min"] == 16
    assert final["metrics"]["last_provider"] == "fake"
    assert final["metrics"]["turns_count"] == 1
    assert final["metrics"]["llm_calls"] >= 1


@pytest.mark.asyncio
async def test_get_session_rehidrata_historial_y_metadata(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "El proyecto se llama Atlas y usaremos Python y FastAPI."},
    )
    await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Equipo de 3 personas. Alcance: MVP en 4 semanas."},
    )

    resp = await async_client.get(f"/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["max_turns"] >= 1
    assert len(body["messages"]) == 4
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
    assert body["project_metadata"]["project_name"] == "Atlas"
    assert body["project_metadata"]["assumed_team_size"] == 3


@pytest.mark.asyncio
async def test_get_session_inexistente_devuelve_404(async_client: AsyncClient) -> None:
    resp = await async_client.get("/sessions/no-existe")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_slash_command_reset_no_llama_llm(
    async_client: AsyncClient,
    fake_llm_stream: dict,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "El proyecto se llama Atlas con Python."},
    )
    stream_calls_before = len(fake_llm_stream["calls"])

    resp = await async_client.post(
        f"/sessions/{sid}/estimate/stream",
        data={"transcript": "/reset"},
    )
    assert resp.status_code == 200
    events = _parse_ndjson(resp.text)
    types = [e["type"] for e in events]
    assert "start" in types
    assert "final" in types
    final = next(e for e in events if e["type"] == "final")
    assert "reiniciada" in final["text"].lower()

    assert len(fake_llm_stream["calls"]) == stream_calls_before
