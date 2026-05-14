"""Tests de render de plantillas v1 (sin APIs externas)."""

import pytest

from app.prompts.loader import render_estimation_prompt
from app.schemas import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _req(
    *,
    description: str = "Descripción de proyecto con al menos veinte caracteres.",
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.PHASES_TABLE,
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )


def test_description_aparece_literal_en_user_prompt() -> None:
    desc = (
        "Texto con saltos\n\tcaracteres especiales: <>& \"comillas\" y emoji 🚀 "
        "para comprobar interpolación literal."
    )
    request = _req(description=desc)
    system_prompt, user_prompt = render_estimation_prompt(request, version="v1")
    assert desc in user_prompt
    assert "## Descripción del proyecto" in user_prompt
    assert desc not in system_prompt


def test_output_format_cambia_system_prompt() -> None:
    r_table = _req(output_format=OutputFormat.PHASES_TABLE)
    r_narrative = _req(output_format=OutputFormat.NARRATIVE)
    sys_table, _ = render_estimation_prompt(r_table, version="v1")
    sys_narr, _ = render_estimation_prompt(r_narrative, version="v1")
    assert "Modo fases" in sys_table
    assert "Modo narrativa" in sys_narr
    assert "Modo narrativa" not in sys_table
    assert "Modo fases" not in sys_narr


def test_detail_level_activa_instrucciones_especificas() -> None:
    r_sum = _req(detail_level=DetailLevel.SUMMARY)
    r_det = _req(detail_level=DetailLevel.DETAILED)
    sys_sum, _ = render_estimation_prompt(r_sum, version="v1")
    sys_det, _ = render_estimation_prompt(r_det, version="v1")
    assert "[NIVEL SUMMARY]" in sys_sum
    assert "[NIVEL DETAILED]" not in sys_sum
    assert "[NIVEL DETAILED]" in sys_det
    assert "[NIVEL SUMMARY]" not in sys_det


def test_version_desconocida_lanza_valueerror() -> None:
    with pytest.raises(ValueError, match="Versión de prompt desconocida"):
        render_estimation_prompt(_req(), version="v99")
