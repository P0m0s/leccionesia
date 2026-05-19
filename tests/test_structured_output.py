"""Tests del parseo de la salida estructurada (JSON validado por Pydantic)."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

from app.structured import parse_structured_response


def test_parse_json_directo() -> None:
    payload = {
        "summary_markdown": "## Plan inicial",
        "line_items": [
            {"name": "Auth", "area": "backend", "t_shirt": "S", "hours_min": 16, "hours_max": 40},
            {"name": "UI listado", "area": "frontend", "t_shirt": "M", "hours_min": 60, "hours_max": 120},
        ],
        "assumptions": [{"text": "Stripe como pasarela"}],
        "risks": [{"text": "Cambios regulatorios", "severity": "medium"}],
        "confidence": 6,
        "next_step": "Confirmar volumen de usuarios",
    }
    est, ok = parse_structured_response(json.dumps(payload))
    assert ok is True
    assert est.line_items[0].t_shirt == "S"
    assert est.confidence == 6

    tmin, tmax = est.computed_totals()
    assert tmin == 16 + 60
    assert tmax == 40 + 120


def test_parse_json_dentro_de_code_fence() -> None:
    text = (
        "Aquí va la estimación:\n\n"
        "```json\n"
        '{"summary_markdown": "OK", "confidence": 4}\n'
        "```\n"
        "Espero que ayude."
    )
    est, ok = parse_structured_response(text)
    assert ok is True
    assert est.summary_markdown == "OK"
    assert est.confidence == 4


def test_parse_markdown_libre_hace_fallback() -> None:
    text = "Esto es solo prosa, no JSON.\n\nPróximo paso: nada."
    est, ok = parse_structured_response(text)
    assert ok is False
    assert est.summary_markdown.startswith("Esto es solo prosa")
    assert est.line_items == []
    assert est.confidence is None


def test_parse_json_invalido_hace_fallback() -> None:
    text = '{"summary_markdown": "roto", "confidence": "no_es_int"}'
    est, ok = parse_structured_response(text)
    assert ok is False
    assert "roto" in est.summary_markdown


def test_parse_json_con_comillas_tipograficas() -> None:
    """El parser debe tolerar smart quotes que algunos modelos devuelven."""
    text = '{\u201csummary_markdown\u201d: \u201cHola\u201d, \u201cconfidence\u201d: 3}'
    est, ok = parse_structured_response(text)
    assert ok is True
    assert est.summary_markdown == "Hola"
    assert est.confidence == 3


def test_parse_json_anidado_en_summary_markdown_se_extrae() -> None:
    """Si el LLM erróneamente mete el JSON entero como string dentro de
    ``summary_markdown``, debemos extraer solo el texto narrativo."""
    inner_json = '{"summary_markdown": "## Plan", "confidence": 4}'
    outer = json.dumps({"summary_markdown": inner_json, "confidence": 5})
    est, ok = parse_structured_response(outer)
    assert ok is True
    assert est.summary_markdown == "## Plan"


@pytest.mark.asyncio
async def test_endpoint_devuelve_structured_ok_cuando_llm_responde_json(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    def _json_builder(messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "summary_markdown": "## Plan v1",
                "line_items": [
                    {"name": "API auth", "area": "backend", "t_shirt": "S", "hours_min": 16, "hours_max": 40},
                ],
                "assumptions": [{"text": "Stack Python + FastAPI"}],
                "risks": [],
                "confidence": 5,
                "next_step": "Confirmar pasarela de pago.",
            },
        )

    fake_llm["response_builder"] = _json_builder

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Necesito estimar un SaaS B2B con auth."},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["structured_ok"] is True
    assert body["text"] == "## Plan v1"
    assert body["structured"]["line_items"][0]["name"] == "API auth"
    assert body["structured"]["confidence"] == 5
    assert body["structured"]["total_hours_min"] == 16
    assert body["structured"]["total_hours_max"] == 40


@pytest.mark.asyncio
async def test_endpoint_estructurado_degradado_mantiene_text(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Aplicación web sencilla."},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["structured_ok"] is False
    assert body["text"]
    assert body["structured"]["summary_markdown"] == body["text"]
    assert body["structured"]["line_items"] == []
