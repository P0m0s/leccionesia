"""
Métricas de stress test para el CAG.

Tres métricas nuevas:
- LatencyBudgetMetric: 1.0 si latency_ms <= budget_ms; 0.0 si no.
- CostBudgetMetric: 1.0 si cost_usd <= budget_usd; 0.0 si no.
- MemoryDriftMetric: 1.0 si el fact aparece en summary, anchors o metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    details: str


class LatencyBudgetMetric:
    """Verifica que la latencia de un turno no exceda un presupuesto."""

    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = budget_ms

    def evaluate(self, latency_ms: float) -> MetricResult:
        passed = latency_ms <= self.budget_ms
        return MetricResult(
            name="LatencyBudget",
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"latency={latency_ms:.1f}ms vs budget={self.budget_ms:.1f}ms",
        )


class CostBudgetMetric:
    """Verifica que el coste acumulado no exceda un presupuesto."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd

    def evaluate(self, cost_usd: float) -> MetricResult:
        passed = cost_usd <= self.budget_usd
        return MetricResult(
            name="CostBudget",
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"cost=${cost_usd:.6f} vs budget=${self.budget_usd:.6f}",
        )


class MemoryDriftMetric:
    """Verifica que un fact clave sigue presente en el contexto del CAG.

    Busca (case-insensitive) en:
    - ``summary``: el ``conversation_summary`` de la sesión.
    - ``anchors``: texto combinado de la metadata (nombre, tipo, tecnologías, alcance).
    - ``metadata_raw``: el dict completo de project_metadata serializado.
    """

    def evaluate(
        self,
        fact: str,
        *,
        summary: str = "",
        anchors: str = "",
        metadata_raw: str = "",
    ) -> MetricResult:
        fact_lower = fact.lower()
        haystack = f"{summary} {anchors} {metadata_raw}".lower()
        found = fact_lower in haystack
        return MetricResult(
            name="MemoryDrift",
            score=1.0 if found else 0.0,
            passed=found,
            details=f"fact='{fact}' {'encontrado' if found else 'NO encontrado'} en contexto ({len(haystack)} chars)",
        )
