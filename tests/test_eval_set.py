"""Tests del eval set: cada brief debe puntuar > 0 con un fake LLM razonable."""

from __future__ import annotations

import json

import pytest

from app.eval.briefs import EVAL_BRIEFS
from app.eval.runner import run_eval


@pytest.fixture
def eval_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sustituye el LLM por una respuesta JSON sintética alineada con cada brief."""

    def _builder(messages: list[dict[str, str]]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        ).lower()

        topics: list[str] = []
        if "móvil" in last_user or "ios" in last_user or "android" in last_user:
            topics.extend(["push", "pago", "reservas", "offline", "sincronización", "checklist"])
        if "saas" in last_user or "multi-tenant" in last_user or "incidencias" in last_user:
            topics.extend(["multi-tenant", "SSO", "suscripción"])
        if "interno" in last_user or "aprobaciones" in last_user or "managers" in last_user:
            topics.extend(["flujo", "SAP", "aprobaciones"])
        if "etl" in last_user or "pipeline" in last_user or "bigquery" in last_user:
            topics.extend(["ETL", "calidad", "pipeline"])

        summary = "## Estimación inicial\n" + " ".join(f"- {t}" for t in set(topics))

        return json.dumps(
            {
                "summary_markdown": summary,
                "line_items": [
                    {"name": "Auth", "area": "backend", "t_shirt": "M", "hours_min": 60, "hours_max": 120},
                    {"name": "UI core", "area": "frontend", "t_shirt": "L", "hours_min": 160, "hours_max": 280},
                    {"name": "Integraciones", "area": "backend", "t_shirt": "M", "hours_min": 80, "hours_max": 200},
                    {"name": "Pruebas y release", "area": "qa", "t_shirt": "M", "hours_min": 80, "hours_max": 200},
                ],
                "phases": [],
                "assumptions": [],
                "risks": [],
                "confidence": 5,
                "next_step": "Confirmar volumen de usuarios.",
            },
        )

    def _fake(messages, *, images=None, metrics_out=None):
        text = _builder(messages)
        if metrics_out is not None:
            metrics_out.update(
                {
                    "model": "gpt-4o-mini",
                    "provider": "fake",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "elapsed_ms": 1.0,
                },
            )
        return text, "gpt-4o-mini", "fake"

    monkeypatch.setattr(
        "app.services.session_service.generate_chat_messages",
        _fake,
    )


def test_eval_runner_da_score_alto_con_fake_llm(eval_llm) -> None:
    results = run_eval()
    assert len(results) == len(EVAL_BRIEFS)
    for r in results:
        assert r.score > 0, f"{r.brief_id} obtuvo score 0"
    avg = sum(r.score for r in results) / len(results)
    assert avg >= 0.5, f"avg_score={avg:.2f} demasiado bajo"


def test_eval_runner_marca_proyecto_correctamente(eval_llm) -> None:
    results = {r.brief_id: r for r in run_eval()}
    assert results["mobile_reservas"].detected_project_type == "mobile_app"
    assert results["data_etl_bigquery"].detected_project_type == "data_pipeline"
    assert results["saas_b2b_incidencias"].detected_project_type == "web_saas"
    assert results["internal_aprobaciones"].detected_project_type == "internal_tool"


def test_brief_dataclass_es_inmutable() -> None:
    b = EVAL_BRIEFS[0]
    with pytest.raises(Exception):
        b.id = "otro"  # type: ignore[misc]
