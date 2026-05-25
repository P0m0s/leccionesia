"""Tests para las métricas de stress test del CAG."""

from __future__ import annotations

from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    MetricResult,
)


# ── LatencyBudgetMetric ──────────────────────────────────────────────────

class TestLatencyBudgetMetric:
    def test_within_budget_passes(self):
        m = LatencyBudgetMetric(budget_ms=5000.0)
        r = m.evaluate(latency_ms=3200.0)
        assert r.passed is True
        assert r.score == 1.0
        assert r.name == "LatencyBudget"

    def test_exceeds_budget_fails(self):
        m = LatencyBudgetMetric(budget_ms=2000.0)
        r = m.evaluate(latency_ms=2500.0)
        assert r.passed is False
        assert r.score == 0.0

    def test_exact_boundary_passes(self):
        m = LatencyBudgetMetric(budget_ms=1000.0)
        r = m.evaluate(latency_ms=1000.0)
        assert r.passed is True
        assert r.score == 1.0


# ── CostBudgetMetric ─────────────────────────────────────────────────────

class TestCostBudgetMetric:
    def test_within_budget_passes(self):
        m = CostBudgetMetric(budget_usd=0.05)
        r = m.evaluate(cost_usd=0.03)
        assert r.passed is True
        assert r.score == 1.0
        assert r.name == "CostBudget"

    def test_exceeds_budget_fails(self):
        m = CostBudgetMetric(budget_usd=0.01)
        r = m.evaluate(cost_usd=0.02)
        assert r.passed is False
        assert r.score == 0.0

    def test_exact_boundary_passes(self):
        m = CostBudgetMetric(budget_usd=0.005)
        r = m.evaluate(cost_usd=0.005)
        assert r.passed is True
        assert r.score == 1.0


# ── MemoryDriftMetric ─────────────────────────────────────────────────────

class TestMemoryDriftMetric:
    def test_fact_in_summary_passes(self):
        m = MemoryDriftMetric()
        r = m.evaluate(
            "Aurora",
            summary="El proyecto Aurora requiere autenticación OAuth2.",
            anchors="",
            metadata_raw="",
        )
        assert r.passed is True
        assert r.score == 1.0
        assert r.name == "MemoryDrift"

    def test_fact_in_anchors_passes(self):
        m = MemoryDriftMetric()
        r = m.evaluate(
            "Flutter",
            summary="",
            anchors="Flutter, Django, PostgreSQL",
            metadata_raw="",
        )
        assert r.passed is True
        assert r.score == 1.0

    def test_fact_in_metadata_passes(self):
        m = MemoryDriftMetric()
        r = m.evaluate(
            "80000",
            summary="",
            anchors="",
            metadata_raw='{"budget": "80000 euros"}',
        )
        assert r.passed is True

    def test_fact_missing_fails(self):
        m = MemoryDriftMetric()
        r = m.evaluate(
            "React Native",
            summary="Usamos Flutter para la app móvil.",
            anchors="Flutter, Django",
            metadata_raw='{"tech": "Flutter"}',
        )
        assert r.passed is False
        assert r.score == 0.0

    def test_case_insensitive_match(self):
        m = MemoryDriftMetric()
        r = m.evaluate(
            "aurora",
            summary="El proyecto AURORA está en marcha.",
            anchors="",
            metadata_raw="",
        )
        assert r.passed is True
        assert r.score == 1.0

    def test_empty_context_fails(self):
        m = MemoryDriftMetric()
        r = m.evaluate("SAP", summary="", anchors="", metadata_raw="")
        assert r.passed is False
        assert r.score == 0.0


# ── MetricResult dataclass ────────────────────────────────────────────────

class TestMetricResult:
    def test_fields(self):
        r = MetricResult(name="Test", score=0.5, passed=True, details="ok")
        assert r.name == "Test"
        assert r.score == 0.5
        assert r.passed is True
        assert r.details == "ok"
