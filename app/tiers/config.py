"""Configuración estática del patrón **tier**.

``TIER_CONFIG`` es la **única fuente de verdad** que enlaza:

- nombre del tier,
- pipeline (handler de orquestación),
- ruta del template Jinja2,
- schema Pydantic de salida,
- modelo del LLM (puede ser distinto al provider por defecto),
- y opcionalmente: herramientas, background, latencia/coste esperados.

Añadir un tier nuevo es una entrada en este dict + un template + un schema.
"""

from __future__ import annotations

from typing import Any, Literal

from app.tiers.schemas import (
    DeveloperEstimate,
    ExecutiveEstimate,
    PmEstimate,
    ResearchEstimate,
)

TierName = Literal["developer", "pm", "executive", "research"]
PipelineName = Literal["single_call", "deep_research"]


TIER_CONFIG: dict[TierName, dict[str, Any]] = {
    "developer": {
        "pipeline": "single_call",
        "template": "tiers/developer.j2",
        "schema": DeveloperEstimate,
        "model": "gpt-4o-mini",
    },
    "pm": {
        "pipeline": "single_call",
        "template": "tiers/pm.j2",
        "schema": PmEstimate,
        "model": "gpt-4o-mini",
    },
    "executive": {
        "pipeline": "single_call",
        "template": "tiers/executive.j2",
        "schema": ExecutiveEstimate,
        "model": "gpt-4o-mini",
    },
    "research": {
        "pipeline": "deep_research",
        "template": "tiers/research.j2",
        "schema": ResearchEstimate,
        # En producción: "o3-deep-research". Mientras tanto, el handler
        # `deep_research` hace una llamada extendida con un modelo capaz.
        "model": "o3-deep-research",
        "tools": ["web_search", "code_interpreter"],
        "background": True,
        "estimated_latency_seconds": 600,
        "estimated_cost_per_call_eur": 5.00,
    },
}


def resolve_tier_config(tier: str) -> dict[str, Any]:
    """Devuelve la configuración de un tier o lanza ``ValueError``."""
    if tier not in TIER_CONFIG:
        msg = (
            f"Tier desconocido: {tier!r}. "
            f"Válidos: {sorted(TIER_CONFIG.keys())}"
        )
        raise ValueError(msg)
    return TIER_CONFIG[tier]  # type: ignore[index]


# Importación tardía para evitar ciclos (pipelines importa schemas/config).
def _build_pipeline_handlers() -> dict[PipelineName, Any]:
    from app.tiers.pipelines import run_deep_research, run_single_call

    return {
        "single_call": run_single_call,
        "deep_research": run_deep_research,
    }


class _LazyHandlers:
    """Mapa lazy: pipelines se importa al primer uso para evitar ciclos."""

    def __init__(self) -> None:
        self._cache: dict[PipelineName, Any] | None = None

    def __getitem__(self, key: PipelineName) -> Any:
        if self._cache is None:
            self._cache = _build_pipeline_handlers()
        return self._cache[key]

    def __contains__(self, key: object) -> bool:
        if self._cache is None:
            self._cache = _build_pipeline_handlers()
        return key in self._cache


PIPELINE_HANDLERS: _LazyHandlers = _LazyHandlers()


__all__ = [
    "PIPELINE_HANDLERS",
    "TIER_CONFIG",
    "PipelineName",
    "TierName",
    "resolve_tier_config",
]
