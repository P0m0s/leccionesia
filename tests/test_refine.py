"""Tests del flag ?refine=true (self-critique de 1 pasada)."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient


def _structured_response(summary: str, confidence: int = 5) -> str:
    return json.dumps(
        {
            "summary_markdown": summary,
            "line_items": [],
            "phases": [],
            "assumptions": [],
            "risks": [],
            "confidence": confidence,
            "next_step": "next",
        },
    )


@pytest.mark.asyncio
async def test_refine_dispara_segunda_llamada_y_devuelve_refined_true(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    sequence = iter([
        _structured_response("## V1: borrador", confidence=4),
        _structured_response("## V2: refinada", confidence=7),
    ])

    def _builder(_messages):
        return next(sequence)

    fake_llm["response_builder"] = _builder

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate?refine=true",
        data={"transcript": "Estimar un SaaS B2B con multi-tenant."},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["refined"] is True
    assert body["text"] == "## V2: refinada"
    assert body["structured"]["confidence"] == 7

    assert len(fake_llm["calls"]) == 2
    second_call = fake_llm["calls"][1]
    last_user = next(m for m in reversed(second_call) if m["role"] == "user")
    assert "Revisa tu propia respuesta" in last_user["content"]


@pytest.mark.asyncio
async def test_sin_refine_solo_una_llamada(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    fake_llm["response_builder"] = lambda _m: _structured_response("solo una")

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]
    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Estimación sencilla."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refined"] is False
    assert len(fake_llm["calls"]) == 1
