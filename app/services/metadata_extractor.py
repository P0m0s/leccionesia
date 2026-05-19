"""
Extracción heurística (regex) de ``ProjectMetadata`` a partir del texto del usuario.

Camino elegido: **heurística simple**. No usa una segunda llamada al LLM, por:

- Determinismo (los tests pueden afirmar cambios concretos sin red ni mocks).
- Coste cero por turno.
- Suficiente para un MVP conversacional; el LLM principal sigue viendo todo el
  diálogo a través de la ventana deslizante, de modo que la metadata es solo un
  "resumen estructurado" inyectado como contexto extra en el system prompt.

Reglas implementadas:

- **project_name**: frases del tipo "el proyecto se llama …", "nombre del proyecto: …".
- **assumed_team_size**: números asociados a "personas", "developers", "team".
- **mentioned_technologies**: catálogo cerrado de tecnologías habituales detectadas
  por coincidencia case-insensitive (Python, FastAPI, React, etc.).
- **agreed_scope**: frases tras "alcance:", "scope:", "vamos a hacer", etc.
"""

from __future__ import annotations

import re

from app.sessions import ProjectMetadata

_TECH_CATALOG: tuple[str, ...] = (
    "Python", "FastAPI", "Django", "Flask",
    "Node.js", "Express", "NestJS",
    "TypeScript", "JavaScript",
    "React", "Next.js", "Vue", "Angular", "Svelte", "Astro",
    "Streamlit",
    "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Elasticsearch",
    "ClickHouse", "BigQuery", "Snowflake", "Redshift",
    "Docker", "Kubernetes",
    "AWS", "GCP", "Azure", "Cloudflare",
    "Stripe", "Twilio", "Adyen",
    "Kafka", "RabbitMQ",
    "GraphQL", "REST",
    "Tailwind",
    "Swift", "Kotlin", "Flutter", "React Native",
    "Rust", "Go", "Bun",
    "OpenAI", "Anthropic", "LangChain",
    "Airflow", "Dagster", "Prefect", "dbt",
)


_PROJECT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mobile_app": (
        "app móvil", "aplicación móvil", "mobile app", "ios", "android",
        "swift", "kotlin", "flutter", "react native",
    ),
    "data_pipeline": (
        "pipeline", "etl", "data warehouse", "ingesta", "ingestion",
        "airflow", "dagster", "dbt", "bigquery", "snowflake",
        "data lake", "lakehouse",
    ),
    "internal_tool": (
        "herramienta interna", "back office", "backoffice", "portal interno",
        "tool interno", "panel de administración", "intranet",
    ),
    "web_saas": (
        "saas", "multi-tenant", "multitenant", "suscripción", "subscription",
        "plan free", "tenants", "portal b2b", "web app",
    ),
}

_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:el\s+proyecto\s+se\s+llama|nombre\s+del\s+proyecto)\s*[:\-]?\s*"
        r"['\"]?([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F0-9_\-]{0,40})['\"]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:project\s+name|project\s+is\s+called)\s*[:\-]?\s*"
        r"['\"]?([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F0-9_\-]{0,40})['\"]?",
        re.IGNORECASE,
    ),
)

_TEAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:equipo|team)\s+de\s+(\d+)\s*(?:personas|devs|developers|ingenieros)?", re.IGNORECASE),
    re.compile(r"(\d+)\s+(?:personas|devs|developers|ingenieros)\s+(?:en\s+el\s+equipo|disponibles|asignad[oa]s)", re.IGNORECASE),
    re.compile(r"team\s+size\s*[:=]\s*(\d+)", re.IGNORECASE),
)

_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:alcance\s+acordado|alcance|scope|agreed\s+scope)\s*[:\-]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"(?:vamos\s+a\s+hacer|construiremos|build)\s+(?:un[ao]?\s+|el\s+)?([^.\n]{10,200})", re.IGNORECASE),
)


def _detect_technologies(text: str) -> list[str]:
    found: list[str] = []
    for tech in _TECH_CATALOG:
        pattern = re.compile(rf"\b{re.escape(tech)}\b", re.IGNORECASE)
        if pattern.search(text) and tech not in found:
            found.append(tech)
    return found


def _detect_first(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip(" '\"")
            if value:
                return value
    return None


def _detect_team_size(text: str) -> int | None:
    for pattern in _TEAM_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _detect_project_type(text: str) -> str | None:
    """Asocia el texto al `project_type` con más keywords coincidentes."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for ptype, keywords in _PROJECT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score:
            scores[ptype] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def extract_metadata_patch(text: str) -> ProjectMetadata:
    """Devuelve una metadata "parche" derivada solo del texto pasado."""
    return ProjectMetadata(
        project_name=_detect_first(_NAME_PATTERNS, text),
        project_type=_detect_project_type(text),
        assumed_team_size=_detect_team_size(text),
        mentioned_technologies=_detect_technologies(text),
        agreed_scope=_detect_first(_SCOPE_PATTERNS, text),
    )


def update_metadata(
    current: ProjectMetadata,
    *texts: str,
) -> ProjectMetadata:
    """
    Actualiza ``current`` con los hallazgos de uno o varios textos.

    Combina sin perder lo previamente acordado: los campos solo se sobrescriben
    si el nuevo turno aporta un valor distinto y no vacío.
    """
    merged = current
    for text in texts:
        if not text:
            continue
        patch = extract_metadata_patch(text)
        merged = merged.merged_with(patch)
    return merged
