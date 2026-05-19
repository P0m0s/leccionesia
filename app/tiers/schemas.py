"""Schemas Pydantic por tier.

Cada tier tiene una estructura **distinta**, no una variante del mismo modelo
con branching. Esto es lo que evita el anti-patrón "schema único con
condicionales" descrito en la lección.

Convención común:

- Los tipos de rango (`HoursRange`, `WeeksRange`, `CostRangeEUR`) sí son
  compartidos: son primitivas geométricas, no la estructura del informe.
- `confidence_level` se usa solo en `executive` y `research`. En
  `developer` la incertidumbre se modela con `uncertainty_drivers` (más
  específico). En `pm` con `blockers`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]
ConfidenceLevel = Literal["low", "medium", "high"]
GoNoGo = Literal["go", "no_go", "conditional_go"]


# ---------------------------------------------------------------------------
# Rangos compartidos (primitivas)
# ---------------------------------------------------------------------------


class HoursRange(BaseModel):
    min: float = Field(ge=0)
    max: float = Field(ge=0)


class WeeksRange(BaseModel):
    min: float = Field(ge=0)
    max: float = Field(ge=0)


class CostRangeEUR(BaseModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Developer — vocabulario técnico, granularidad alta, drivers de incertidumbre
# ---------------------------------------------------------------------------


class DevComponent(BaseModel):
    """Una pieza técnica concreta del sistema."""

    name: str
    description: str
    hours_range: HoursRange
    complexity: Severity


class TechnicalRisk(BaseModel):
    description: str
    mitigation: str | None = None


class DeveloperEstimate(BaseModel):
    """Estimación con vocabulario técnico: componentes, stack, incertidumbres."""

    components: list[DevComponent]
    technical_risks: list[TechnicalRisk]
    stack_assumptions: list[str]
    uncertainty_drivers: list[str]
    total_hours_range: HoursRange


# ---------------------------------------------------------------------------
# PM — fases, hitos, composición de equipo, blockers
# ---------------------------------------------------------------------------


class Phase(BaseModel):
    name: str
    duration_weeks: WeeksRange
    deliverables: list[str]
    dependencies: list[str] = Field(default_factory=list)


class Milestone(BaseModel):
    name: str
    week: int = Field(ge=0)
    deliverables: list[str]


class TeamRole(BaseModel):
    role: str
    fte: float = Field(ge=0, le=10)
    weeks: float = Field(ge=0)


class Blocker(BaseModel):
    description: str
    impact: Severity


class PmEstimate(BaseModel):
    """Estimación con vocabulario de gestión: fases, hitos, equipo, blockers."""

    phases: list[Phase]
    milestones: list[Milestone]
    team_composition: list[TeamRole]
    duration_weeks_range: WeeksRange
    blockers: list[Blocker]


# ---------------------------------------------------------------------------
# Executive — coste/duración top-level, sin jerga técnica, go/no-go
# ---------------------------------------------------------------------------


class ExecutiveRisk(BaseModel):
    """Riesgo expresado en lenguaje de negocio, sin jerga técnica."""

    headline: str
    impact: str  # narrativa, no escala técnica


class ExecutiveEstimate(BaseModel):
    """Estimación ejecutiva: 1 página, coste/plazo headline, recomendación."""

    headline_cost_range: CostRangeEUR
    headline_duration_range: WeeksRange
    confidence_level: ConfidenceLevel
    top_three_risks: list[ExecutiveRisk] = Field(min_length=0, max_length=3)
    go_no_go_recommendation: GoNoGo
    rationale: str = Field(
        description="Una a tres frases que justifican la recomendación.",
    )


# ---------------------------------------------------------------------------
# Research — informe largo con índice, secciones y citas (deep_research)
# ---------------------------------------------------------------------------


class ResearchCitation(BaseModel):
    source: str
    quote: str | None = None
    url: str | None = None


class ResearchSection(BaseModel):
    title: str
    content_markdown: str
    citations: list[ResearchCitation] = Field(default_factory=list)


class ResearchEstimate(BaseModel):
    """Informe de investigación profundo. Pipeline ``deep_research``."""

    title: str
    executive_summary: str
    table_of_contents: list[str]
    sections: list[ResearchSection]
    methodology: str
    citations: list[ResearchCitation] = Field(default_factory=list)
    total_hours_range: HoursRange
    total_cost_range: CostRangeEUR
    confidence_level: ConfidenceLevel
    next_steps: list[str]


__all__ = [
    "Blocker",
    "ConfidenceLevel",
    "CostRangeEUR",
    "DevComponent",
    "DeveloperEstimate",
    "ExecutiveEstimate",
    "ExecutiveRisk",
    "GoNoGo",
    "HoursRange",
    "Milestone",
    "Phase",
    "PmEstimate",
    "ResearchCitation",
    "ResearchEstimate",
    "ResearchSection",
    "Severity",
    "TeamRole",
    "TechnicalRisk",
    "WeeksRange",
]
