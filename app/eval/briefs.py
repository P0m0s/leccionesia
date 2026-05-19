"""
Eval set de briefs canónicos.

Cada brief define:

- ``id``: identificador estable.
- ``brief``: texto que el usuario enviaría como primer turno.
- ``expected_project_type``: ``project_type`` que la heurística debería detectar.
- ``expected_total_hours_range``: rango ``(min, max)`` razonable para el total
  agregado. Si la estimación del LLM cae fuera de este rango, queda marcada.
- ``required_techs``: tecnologías que **deberían** aparecer en
  ``mentioned_technologies`` (al menos una de ellas).
- ``required_topics``: subcadenas que **deberían** aparecer en el
  ``summary_markdown`` (mínimo una).

Esto sirve como red de seguridad ligera para detectar regresiones cuando se
modifican prompts (chat/v1, chat/v2, examples).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Brief:
    id: str
    brief: str
    expected_project_type: str | None = None
    expected_total_hours_range: tuple[float, float] | None = None
    required_techs: tuple[str, ...] = field(default_factory=tuple)
    required_topics: tuple[str, ...] = field(default_factory=tuple)


EVAL_BRIEFS: tuple[Brief, ...] = (
    Brief(
        id="mobile_reservas",
        brief=(
            "MVP de app móvil iOS y Android para reservas de servicios, "
            "con pago con tarjeta y notificaciones push. Equipo de 3 personas."
        ),
        expected_project_type="mobile_app",
        expected_total_hours_range=(400, 1200),
        required_techs=("Swift", "Kotlin", "Flutter", "React Native", "Stripe"),
        required_topics=("push", "pago", "reservas"),
    ),
    Brief(
        id="saas_b2b_incidencias",
        brief=(
            "Plataforma SaaS B2B multi-tenant para gestión de incidencias, "
            "con SSO, suscripción Stripe y reporting. Equipo de 4 personas."
        ),
        expected_project_type="web_saas",
        expected_total_hours_range=(600, 1800),
        required_techs=("Stripe",),
        required_topics=("multi-tenant", "SSO", "suscripción"),
    ),
    Brief(
        id="internal_aprobaciones",
        brief=(
            "Portal interno para que managers aprueben gastos con flujo "
            "multi-nivel y exportación contable a SAP."
        ),
        expected_project_type="internal_tool",
        expected_total_hours_range=(200, 800),
        required_topics=("flujo", "SAP", "aprob"),
    ),
    Brief(
        id="data_etl_bigquery",
        brief=(
            "Pipeline ETL incremental desde ERP legado hacia BigQuery con "
            "validaciones de calidad y alertas. Orquestado con Airflow."
        ),
        expected_project_type="data_pipeline",
        expected_total_hours_range=(300, 1000),
        required_techs=("Airflow", "BigQuery"),
        required_topics=("ETL", "calidad", "pipeline"),
    ),
    Brief(
        id="mobile_offline",
        brief=(
            "App iOS y Android offline-first para técnicos de campo con "
            "checklist, fotos y sincronización diferida. Backend en FastAPI + PostgreSQL."
        ),
        expected_project_type="mobile_app",
        expected_total_hours_range=(500, 1500),
        required_techs=("FastAPI", "PostgreSQL"),
        required_topics=("offline", "sincronización", "checklist"),
    ),
)
