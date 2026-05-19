"""Tests del few-shot dinámico por project_type."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.prompts.loader import render_chat_system_prompt
from app.services.metadata_extractor import _detect_project_type
from app.sessions import ProjectMetadata


def test_detector_project_type_mobile() -> None:
    assert _detect_project_type("Quiero una app móvil iOS y Android") == "mobile_app"


def test_detector_project_type_data() -> None:
    assert _detect_project_type("Necesito un pipeline ETL hacia BigQuery") == "data_pipeline"


def test_detector_project_type_saas() -> None:
    assert _detect_project_type("Plataforma SaaS multi-tenant con suscripción") == "web_saas"


def test_detector_project_type_internal() -> None:
    assert _detect_project_type("Herramienta interna de aprobación con back office") == "internal_tool"


def test_system_prompt_incluye_examples_por_project_type() -> None:
    pm_mobile = ProjectMetadata(project_type="mobile_app")
    rendered_mobile = render_chat_system_prompt(pm_mobile)
    assert "dominio **mobile_app**" in rendered_mobile
    assert "Apple/Play" in rendered_mobile

    pm_data = ProjectMetadata(project_type="data_pipeline")
    rendered_data = render_chat_system_prompt(pm_data)
    assert "dominio **data_pipeline**" in rendered_data
    assert "BigQuery" in rendered_data or "Snowflake" in rendered_data


def test_system_prompt_default_si_project_type_desconocido() -> None:
    pm_none = ProjectMetadata()
    rendered_default = render_chat_system_prompt(pm_none)
    assert "Ejemplos de referencia" in rendered_default
    assert "dominio **mobile_app**" not in rendered_default


def test_system_prompt_incluye_calibracion_tshirt_y_schema_json() -> None:
    rendered = render_chat_system_prompt(ProjectMetadata())
    assert "T-shirt sizing" in rendered
    assert "**XS**" in rendered
    assert "summary_markdown" in rendered
    assert "line_items" in rendered
    assert "confidence" in rendered


@pytest.mark.asyncio
async def test_project_type_se_detecta_y_persiste(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]
    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Quiero estimar un pipeline ETL incremental hacia BigQuery."},
    )
    assert resp.status_code == 200, resp.text
    metadata = resp.json()["project_metadata"]
    assert metadata["project_type"] == "data_pipeline"
