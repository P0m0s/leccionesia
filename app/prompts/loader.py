"""Carga y render de plantillas Jinja2 para estimación (CAG versionado)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest

_PROMPTS_ROOT = Path(__file__).resolve().parent
_VALID_VERSIONS = frozenset({"v1", "v2"})


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_ROOT)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """
    Devuelve (system_prompt, user_prompt) para la versión de plantilla indicada.
    """
    if version not in _VALID_VERSIONS:
        msg = f"Versión de prompt desconocida: {version!r} (válidas: {sorted(_VALID_VERSIONS)})"
        raise ValueError(msg)

    env = _environment()
    refs = list(request.reference_projects) if request.reference_projects else []

    ctx = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "reference_projects": refs,
    }

    system_t = env.get_template(f"estimation/{version}/system.j2")
    user_t = env.get_template(f"estimation/{version}/user.j2")
    return system_t.render(**ctx), user_t.render(**ctx)
