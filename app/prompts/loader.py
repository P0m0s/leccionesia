"""Carga y render de plantillas Jinja2 para estimación (CAG versionado)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest

if TYPE_CHECKING:
    from app.sessions import ProjectMetadata

_PROMPTS_ROOT = Path(__file__).resolve().parent
_VALID_VERSIONS = frozenset({"v1", "v2"})
_VALID_CHAT_VERSIONS = frozenset({"v1", "v2"})


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


def _format_project_metadata(project_metadata: ProjectMetadata | None) -> str:
    """Formatea la metadata para inyectarla en el bloque ``<project_metadata>``.

    Si todos los campos están vacíos devuelve cadena vacía (el bloque queda
    abierto-cerrado sin contenido, tal como pide el ejercicio).
    """
    if project_metadata is None:
        return ""

    lines: list[str] = []
    if project_metadata.project_name:
        lines.append(f"- project_name: {project_metadata.project_name}")
    if project_metadata.project_type:
        lines.append(f"- project_type: {project_metadata.project_type}")
    if project_metadata.assumed_team_size is not None:
        lines.append(f"- assumed_team_size: {project_metadata.assumed_team_size}")
    if project_metadata.mentioned_technologies:
        techs = ", ".join(project_metadata.mentioned_technologies)
        lines.append(f"- mentioned_technologies: {techs}")
    if project_metadata.agreed_scope:
        lines.append(f"- agreed_scope: {project_metadata.agreed_scope}")
    if project_metadata.conversation_summary:
        lines.append("- conversation_summary:")
        lines.append(f"    {project_metadata.conversation_summary}")
    return "\n".join(lines)


_VALID_EXAMPLES = frozenset({"default", "mobile_app", "web_saas", "internal_tool", "data_pipeline"})


def _render_examples_block(env: Environment, version: str, project_type: str | None) -> str:
    """Devuelve el bloque few-shot apropiado para el project_type detectado.

    Si la versión actual no tiene su carpeta ``examples/``, hace fallback a la
    de ``chat/v1/examples/`` (compartir ejemplos por defecto).
    """
    key = project_type if project_type in _VALID_EXAMPLES else "default"
    candidates = [f"chat/{version}/examples/{key}.j2", f"chat/v1/examples/{key}.j2"]
    from jinja2 import TemplateNotFound

    for path in candidates:
        try:
            template = env.get_template(path)
        except TemplateNotFound:
            continue
        return template.render()
    return ""


def render_chat_refine_prompt(version: str = "v1") -> str:
    """Render del prompt de auto-crítica (self-critique) para el flag ``refine``."""
    if version not in _VALID_CHAT_VERSIONS:
        msg = (
            f"Versión de prompt de chat desconocida: {version!r} "
            f"(válidas: {sorted(_VALID_CHAT_VERSIONS)})"
        )
        raise ValueError(msg)
    env = _environment()
    return env.get_template(f"chat/{version}/refine.j2").render()


def render_chat_system_prompt(
    project_metadata: ProjectMetadata | None,
    version: str = "v1",
) -> str:
    """Renderiza el system prompt conversacional inyectando ``project_metadata``.

    Elige el bloque few-shot según ``project_metadata.project_type`` y añade las
    instrucciones de salida estructurada (JSON) compartidas.
    """
    if version not in _VALID_CHAT_VERSIONS:
        msg = (
            f"Versión de prompt de chat desconocida: {version!r} "
            f"(válidas: {sorted(_VALID_CHAT_VERSIONS)})"
        )
        raise ValueError(msg)

    from app.structured import STRUCTURED_OUTPUT_INSTRUCTIONS

    env = _environment()
    template = env.get_template(f"chat/{version}/system.j2")
    ptype = project_metadata.project_type if project_metadata else None
    examples_block = _render_examples_block(env, version, ptype)
    return template.render(
        project_metadata=_format_project_metadata(project_metadata),
        examples_block=examples_block,
        structured_output_instructions=STRUCTURED_OUTPUT_INSTRUCTIONS,
    )
