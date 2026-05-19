"""Tests de la acumulación de métricas y coste estimado por sesión."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.services.cost import estimate_cost_usd


def test_cost_modelo_conocido() -> None:
    cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    assert cost is not None
    assert cost == pytest.approx(0.00015 + 0.0006 * 0.5, rel=1e-9)


def test_cost_modelo_desconocido_devuelve_none() -> None:
    assert estimate_cost_usd("unknown-model", 100, 100) is None


def test_cost_tokens_none_devuelve_none() -> None:
    assert estimate_cost_usd("gpt-4o-mini", None, 100) is None


@pytest.mark.asyncio
async def test_metricas_se_acumulan_entre_turnos(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    for i in range(3):
        resp = await async_client.post(
            f"/sessions/{sid}/estimate",
            data={"transcript": f"Turno {i}."},
        )
        assert resp.status_code == 200

    body = resp.json()
    metrics = body["metrics"]
    assert metrics["turns_count"] == 3
    assert metrics["llm_calls"] == 3
    assert metrics["input_tokens_total"] == 300
    assert metrics["output_tokens_total"] > 0
    assert metrics["estimated_cost_usd"] > 0
    assert metrics["last_provider"] == "fake"
    assert metrics["last_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_metricas_refine_cuenta_doble(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]
    resp = await async_client.post(
        f"/sessions/{sid}/estimate?refine=true",
        data={"transcript": "Estimar"},
    )
    assert resp.status_code == 200
    metrics = resp.json()["metrics"]
    assert metrics["turns_count"] == 1
    assert metrics["llm_calls"] == 2
