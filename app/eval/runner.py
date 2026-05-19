"""
Runner del eval set.

Ejecuta cada brief contra la sesión conversacional y produce un informe con:

- detección de ``project_type`` correcta o no,
- tecnologías esperadas presentes en la metadata,
- temas requeridos cubiertos en el ``summary_markdown``,
- total de horas dentro del rango razonable,
- confidence reportada por el modelo.

Salida JSON-friendly. Pensado para ejecutarse manualmente desde CLI o desde un
pipeline de CI cuando hay credenciales LLM disponibles. **No se ejecuta en los
tests unitarios** porque hace llamadas reales.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from app.eval.briefs import EVAL_BRIEFS, Brief
from app.services.session_service import run_session_turn
from app.sessions import Session
from app.structured import StructuredEstimation


@dataclass
class BriefResult:
    brief_id: str
    project_type_ok: bool
    detected_project_type: str | None
    required_techs_present: list[str]
    required_techs_missing: list[str]
    required_topics_present: list[str]
    required_topics_missing: list[str]
    total_hours_min: float | None
    total_hours_max: float | None
    total_in_range: bool | None
    confidence: int | None
    summary_excerpt: str

    @property
    def score(self) -> float:
        """Score 0..1 simple para resumir el brief."""
        components: list[float] = [1.0 if self.project_type_ok else 0.0]
        if self.required_techs_present or self.required_techs_missing:
            components.append(
                len(self.required_techs_present)
                / (len(self.required_techs_present) + len(self.required_techs_missing)),
            )
        if self.required_topics_present or self.required_topics_missing:
            components.append(
                len(self.required_topics_present)
                / (len(self.required_topics_present) + len(self.required_topics_missing)),
            )
        if self.total_in_range is not None:
            components.append(1.0 if self.total_in_range else 0.0)
        return sum(components) / len(components) if components else 0.0


def _evaluate_brief(brief: Brief, session: Session) -> BriefResult:
    turn = run_session_turn(
        session=session,
        transcript=brief.brief,
        attachments=[],
    )
    estimation: StructuredEstimation = turn.structured
    summary = turn.text or ""

    detected = session.project_metadata.project_type
    project_type_ok = (
        brief.expected_project_type is None
        or detected == brief.expected_project_type
    )

    mentioned = set(session.project_metadata.mentioned_technologies)
    techs_present = [t for t in brief.required_techs if t in mentioned]
    techs_missing = list(brief.required_techs)
    if brief.required_techs:
        techs_missing = (
            [] if any(t in mentioned for t in brief.required_techs) else list(brief.required_techs)
        )

    summary_lower = summary.lower()
    topics_present = [t for t in brief.required_topics if t.lower() in summary_lower]
    topics_missing = [t for t in brief.required_topics if t.lower() not in summary_lower]

    tmin, tmax = estimation.computed_totals()
    in_range: bool | None
    if brief.expected_total_hours_range is None or tmin is None or tmax is None:
        in_range = None
    else:
        emin, emax = brief.expected_total_hours_range
        in_range = (tmax >= emin) and (tmin <= emax)

    return BriefResult(
        brief_id=brief.id,
        project_type_ok=project_type_ok,
        detected_project_type=detected,
        required_techs_present=techs_present,
        required_techs_missing=techs_missing,
        required_topics_present=topics_present,
        required_topics_missing=topics_missing,
        total_hours_min=tmin,
        total_hours_max=tmax,
        total_in_range=in_range,
        confidence=estimation.confidence,
        summary_excerpt=summary[:240],
    )


def run_eval(briefs: tuple[Brief, ...] = EVAL_BRIEFS) -> list[BriefResult]:
    """Ejecuta todos los briefs y devuelve los resultados."""
    results: list[BriefResult] = []
    for brief in briefs:
        session = Session(session_id=f"eval-{brief.id}")
        results.append(_evaluate_brief(brief, session))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta el eval set del Estimador.")
    parser.add_argument("--out", help="Ruta de salida JSON (opcional).")
    args = parser.parse_args()

    results = run_eval()
    payload = {
        "n_briefs": len(results),
        "avg_score": (
            sum(r.score for r in results) / len(results) if results else 0.0
        ),
        "results": [asdict(r) for r in results],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(serialized)
        print(f"Eval escrita en {args.out}. avg_score={payload['avg_score']:.2f}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
