from enum import Enum

from pydantic import BaseModel, Field


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[str] | None = Field(
        default=None,
        description="Proyectos de referencia (usado en plantillas v2).",
    )


class EstimationResponse(BaseModel):
    text: str
    prompt_version: str


class SessionCreateResponse(BaseModel):
    session_id: str


class SessionEstimateResponse(BaseModel):
    session_id: str
    text: str = Field(
        description="Resumen markdown (compatibilidad). Equivale a structured.summary_markdown.",
    )
    structured: dict | None = Field(
        default=None,
        description="Estimación estructurada cuando el LLM devolvió JSON parseable.",
    )
    structured_ok: bool = Field(
        default=False,
        description="True si el LLM devolvió un JSON válido conforme al schema.",
    )
    refined: bool = Field(
        default=False,
        description="True si el turno pasó por la fase de auto-crítica (`refine`).",
    )
    tier: str | None = Field(
        default=None,
        description=(
            "Tier que se usó para construir la respuesta (developer, pm, executive, "
            "research) o `None` si se usó el flujo conversacional clásico."
        ),
    )
    pipeline: str | None = Field(
        default=None,
        description="Pipeline ejecutado (`single_call` o `deep_research`).",
    )
    project_metadata: dict
    metrics: dict = Field(default_factory=dict)
    attachments_processed: list[str] = Field(default_factory=list)
    attachments_failed: list[dict] = Field(default_factory=list)


class SessionStateResponse(BaseModel):
    """Estado completo de una sesión para rehidratar UIs."""

    session_id: str
    project_metadata: dict
    messages: list[dict]
    max_turns: int
    metrics: dict = Field(default_factory=dict)
