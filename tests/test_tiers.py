"""Tests del patrón **tier**.

Cubren:
- Dependency `get_caller_context` (Opción B con headers).
- TIER_CONFIG y `resolve_tier_config`.
- Schemas Pydantic por tier (estructuras distintas).
- Pipelines `single_call` y `deep_research` end-to-end vía endpoint.
- Compat: si no se envía header de tier, el endpoint cae al flujo clásico.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from app.tiers import (
    PIPELINE_HANDLERS,
    TIER_CONFIG,
    CallerContext,
    DeveloperEstimate,
    ExecutiveEstimate,
    PmEstimate,
    ResearchEstimate,
    resolve_tier_config,
)

# ---------------------------------------------------------------------------
# Unit tests: config, schemas, dependency
# ---------------------------------------------------------------------------


def test_tier_config_completo() -> None:
    assert set(TIER_CONFIG.keys()) == {"developer", "pm", "executive", "research"}
    for tier, cfg in TIER_CONFIG.items():
        assert "pipeline" in cfg, tier
        assert "template" in cfg, tier
        assert "schema" in cfg, tier
        assert "model" in cfg, tier
        assert cfg["pipeline"] in {"single_call", "deep_research"}, tier


def test_resolve_tier_config_valida_tier() -> None:
    cfg = resolve_tier_config("developer")
    assert cfg["schema"] is DeveloperEstimate
    assert cfg["pipeline"] == "single_call"

    with pytest.raises(ValueError, match="Tier desconocido"):
        resolve_tier_config("ceo")


def test_pipeline_handlers_dispatch() -> None:
    # No instancia: solo confirma que ambas pipelines están registradas.
    assert "single_call" in PIPELINE_HANDLERS
    assert "deep_research" in PIPELINE_HANDLERS


def test_research_marcado_como_background_y_caro() -> None:
    cfg = resolve_tier_config("research")
    assert cfg["pipeline"] == "deep_research"
    assert cfg["background"] is True
    assert cfg["estimated_latency_seconds"] > 0
    assert cfg["estimated_cost_per_call_eur"] > 0
    assert "web_search" in cfg["tools"]


def test_caller_context_pydantic() -> None:
    c = CallerContext(user_id="alice", tier="pm")
    assert c.user_id == "alice"
    assert c.tier == "pm"
    with pytest.raises(ValidationError):
        CallerContext(user_id="bob", tier="ceo")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schemas: estructuras genuinamente distintas (anti-anti-patrón 2)
# ---------------------------------------------------------------------------


def test_schemas_son_distintos_no_un_alias() -> None:
    dev_fields = set(DeveloperEstimate.model_fields.keys())
    pm_fields = set(PmEstimate.model_fields.keys())
    exec_fields = set(ExecutiveEstimate.model_fields.keys())
    research_fields = set(ResearchEstimate.model_fields.keys())

    assert "components" in dev_fields
    assert "components" not in pm_fields
    assert "phases" in pm_fields
    assert "phases" not in dev_fields
    assert "headline_cost_range" in exec_fields
    assert "headline_cost_range" not in pm_fields
    assert "sections" in research_fields
    assert "sections" not in exec_fields


# ---------------------------------------------------------------------------
# Endpoint: cada tier responde con su schema
# ---------------------------------------------------------------------------


def _dev_payload() -> dict[str, Any]:
    return {
        "components": [
            {
                "name": "API Auth",
                "description": "Endpoint OAuth2 + JWT",
                "hours_range": {"min": 16, "max": 40},
                "complexity": "medium",
            },
            {
                "name": "ETL diario",
                "description": "Job nocturno de ingesta",
                "hours_range": {"min": 24, "max": 60},
                "complexity": "high",
            },
            {
                "name": "Dashboard React",
                "description": "Vista de KPIs",
                "hours_range": {"min": 40, "max": 100},
                "complexity": "medium",
            },
        ],
        "technical_risks": [
            {"description": "Modelo de datos del cliente desconocido", "mitigation": "Discovery 1 semana"},
        ],
        "stack_assumptions": ["FastAPI", "PostgreSQL", "React"],
        "uncertainty_drivers": ["Volumen real de tráfico", "Integración con SAP"],
        "total_hours_range": {"min": 80, "max": 200},
    }


def _pm_payload() -> dict[str, Any]:
    return {
        "phases": [
            {
                "name": "Discovery",
                "duration_weeks": {"min": 1, "max": 2},
                "deliverables": ["BRD", "Arquitectura"],
                "dependencies": [],
            },
            {
                "name": "Build",
                "duration_weeks": {"min": 6, "max": 10},
                "deliverables": ["MVP"],
                "dependencies": ["Discovery"],
            },
            {
                "name": "Hardening",
                "duration_weeks": {"min": 2, "max": 3},
                "deliverables": ["UAT", "Pase a producción"],
                "dependencies": ["Build"],
            },
        ],
        "milestones": [
            {"name": "Kickoff", "week": 0, "deliverables": ["Plan firmado"]},
            {"name": "Go-Live", "week": 12, "deliverables": ["Producción"]},
        ],
        "team_composition": [
            {"role": "Tech Lead", "fte": 0.5, "weeks": 12},
            {"role": "Backend dev", "fte": 1.0, "weeks": 10},
        ],
        "duration_weeks_range": {"min": 9, "max": 15},
        "blockers": [
            {"description": "Decisión sobre proveedor de auth", "impact": "high"},
        ],
    }


def _exec_payload() -> dict[str, Any]:
    return {
        "headline_cost_range": {"min": 20000, "max": 45000},
        "headline_duration_range": {"min": 10, "max": 16},
        "confidence_level": "medium",
        "top_three_risks": [
            {"headline": "Dependencia de proveedor externo", "impact": "Puede retrasar 3-4 semanas."},
            {"headline": "Cambio regulatorio", "impact": "Replantearía el alcance del módulo de cumplimiento."},
        ],
        "go_no_go_recommendation": "conditional_go",
        "rationale": "Negocio justifica la inversión si se cierra el contrato con el proveedor de identidad.",
    }


def _research_payload() -> dict[str, Any]:
    return {
        "title": "Estimación de plataforma de telemedicina",
        "executive_summary": "Plataforma viable en 14-22 semanas, coste 45-90k€, confianza media.",
        "table_of_contents": ["Contexto", "Arquitectura", "Estimación", "Riesgos"],
        "sections": [
            {
                "title": "Contexto",
                "content_markdown": "El cliente atiende 12 clínicas privadas...",
                "citations": [],
            },
            {
                "title": "Arquitectura",
                "content_markdown": "Proponemos microservicios sobre AWS con FHIR como estándar...",
                "citations": [
                    {"source": "HL7 FHIR Spec", "quote": None, "url": "https://hl7.org/fhir"},
                ],
            },
            {
                "title": "Estimación",
                "content_markdown": "Esfuerzo total estimado: 720-1440 h...",
                "citations": [],
            },
            {
                "title": "Riesgos",
                "content_markdown": "Cumplimiento HIPAA / RGPD es el riesgo dominante...",
                "citations": [],
            },
        ],
        "methodology": "Analogía con 3 proyectos previos del sector salud + benchmarks públicos.",
        "citations": [{"source": "Informe sector salud 2024", "quote": None, "url": None}],
        "total_hours_range": {"min": 720, "max": 1440},
        "total_cost_range": {"min": 43200, "max": 86400},
        "confidence_level": "medium",
        "next_steps": ["Cerrar proveedor EHR", "Definir SLA de telecónsultas", "Reunión clínicas piloto"],
    }


_TIER_PAYLOADS: dict[str, dict[str, Any]] = {
    "developer": _dev_payload(),
    "pm": _pm_payload(),
    "executive": _exec_payload(),
    "research": _research_payload(),
}


@pytest.mark.parametrize("tier", ["developer", "pm", "executive", "research"])
@pytest.mark.asyncio
async def test_endpoint_devuelve_estructura_propia_por_tier(
    tier: str,
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    fake_llm["response_builder"] = lambda _msgs: json.dumps(_TIER_PAYLOADS[tier])

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Estima el proyecto descrito en el diálogo."},
        headers={"X-Estimator-Tier": tier, "X-Estimator-User": "tester"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["tier"] == tier
    expected_pipeline = "deep_research" if tier == "research" else "single_call"
    assert body["pipeline"] == expected_pipeline
    assert body["structured_ok"] is True

    expected_keys = set(_TIER_PAYLOADS[tier].keys())
    received_keys = set(body["structured"].keys())
    assert expected_keys.issubset(received_keys), (
        f"Faltan claves en tier {tier}: {expected_keys - received_keys}"
    )


@pytest.mark.asyncio
async def test_endpoint_sin_header_de_tier_usa_flujo_clasico(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    """Compat: si el cliente no envía X-Estimator-Tier, no se activa el tier."""
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Una app móvil con auth y pagos."},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["tier"] is None
    assert body["pipeline"] is None
    # El schema clásico tiene `line_items`, `phases`, `confidence`...
    assert "line_items" in body["structured"]


@pytest.mark.asyncio
async def test_endpoint_tier_invalido_devuelve_400(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "x"},
        headers={"X-Estimator-Tier": "ceo"},
    )
    assert resp.status_code == 400
    assert "ceo" in resp.json()["detail"].lower() or "tier" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_endpoint_tier_json_invalido_devuelve_422(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    """Si el modelo devuelve algo que no se valida contra el schema del tier,
    el endpoint responde 422 explícitamente en lugar de fallar silencioso."""
    fake_llm["response_builder"] = lambda _msgs: "Esto no es JSON válido"

    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "x"},
        headers={"X-Estimator-Tier": "developer"},
    )
    assert resp.status_code == 422
    assert "DeveloperEstimate" in resp.json()["detail"] or "JSON" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_tier_mantiene_memoria_conversacional(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    """Cambiar de tier dentro de la misma sesión preserva el historial."""
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    fake_llm["response_builder"] = lambda _msgs: json.dumps(_dev_payload())
    await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Proyecto Atlas con Python y FastAPI."},
        headers={"X-Estimator-Tier": "developer"},
    )

    fake_llm["response_builder"] = lambda _msgs: json.dumps(_exec_payload())
    r = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Ahora preséntalo al comité ejecutivo."},
        headers={"X-Estimator-Tier": "executive"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "executive"

    state = await async_client.get(f"/sessions/{sid}")
    msgs = state.json()["messages"]
    # 2 turnos = 4 mensajes (user/assistant × 2).
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user"
    assert msgs[2]["role"] == "user"
    assert "Atlas" in msgs[0]["content"] or "Atlas" in state.json()["project_metadata"].get("project_name", "")
