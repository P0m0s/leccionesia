"""
Utilidades compartidas por el frontend Streamlit.

Cubren:

- Render de ``project_metadata`` como tarjetas (3.1).
- Extracción de tablas markdown para convertirlas a ``st.dataframe`` (3.4).
- Exportación de la conversación a Markdown / PDF (2.3).
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any

PROJECT_TYPE_ICONS: dict[str, str] = {
    "mobile_app": "📱",
    "web_saas": "🌐",
    "internal_tool": "🛠️",
    "data_pipeline": "🧬",
}

METADATA_FIELD_LABELS: dict[str, str] = {
    "project_name": "Proyecto",
    "project_type": "Tipo",
    "assumed_team_size": "Equipo",
    "mentioned_technologies": "Tecnologías",
    "agreed_scope": "Alcance acordado",
    "conversation_summary": "Resumen previo",
}


def metadata_card_specs(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Convierte un ``project_metadata`` en una lista de "tarjetas" planas."""
    cards: list[dict[str, str]] = []
    if metadata.get("project_name"):
        cards.append({"icon": "📌", "label": "Proyecto", "value": metadata["project_name"]})
    if metadata.get("project_type"):
        icon = PROJECT_TYPE_ICONS.get(metadata["project_type"], "🗂️")
        cards.append({"icon": icon, "label": "Tipo", "value": metadata["project_type"]})
    if metadata.get("assumed_team_size") is not None:
        cards.append(
            {
                "icon": "👥",
                "label": "Equipo",
                "value": f"{metadata['assumed_team_size']} personas",
            },
        )
    techs = metadata.get("mentioned_technologies") or []
    if techs:
        cards.append({"icon": "🧪", "label": "Tecnologías", "value": ", ".join(techs)})
    if metadata.get("agreed_scope"):
        cards.append({"icon": "🎯", "label": "Alcance", "value": metadata["agreed_scope"]})
    if metadata.get("conversation_summary"):
        cards.append(
            {"icon": "📚", "label": "Resumen previo", "value": metadata["conversation_summary"]},
        )
    return cards


_MD_TABLE_RE = re.compile(
    r"((?:^\|.+\|\s*$\n){2,}(?:^\|.+\|\s*$\n?)*)",
    re.MULTILINE,
)


def parse_markdown_tables(text: str) -> list[list[list[str]]]:
    """Extrae tablas markdown como listas ``[[fila1], [fila2], …]``.

    Las separadoras (``|---|---|``) se descartan. La primera fila es cabecera.
    """
    out: list[list[list[str]]] = []
    for match in _MD_TABLE_RE.finditer(text):
        block = match.group(1).strip().splitlines()
        rows: list[list[str]] = []
        for raw in block:
            line = raw.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c or "") for c in cells if c):
                continue
            rows.append(cells)
        if len(rows) >= 2:
            out.append(rows)
    return out


def conversation_to_markdown(
    session_id: str | None,
    history: list[dict[str, str]],
    metadata: dict[str, Any] | None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Serializa una conversación a un Markdown listo para descargar."""
    parts: list[str] = []
    parts.append(f"# Conversación Estimador — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if session_id:
        parts.append(f"_session_id_: `{session_id}`\n")
    if metadata:
        parts.append("## project_metadata")
        for k, v in metadata.items():
            if v in (None, "", []):
                continue
            if isinstance(v, list):
                v = ", ".join(v)
            parts.append(f"- **{METADATA_FIELD_LABELS.get(k, k)}**: {v}")
        parts.append("")
    if metrics:
        parts.append("## Métricas")
        for k in (
            "turns_count",
            "llm_calls",
            "input_tokens_total",
            "output_tokens_total",
            "estimated_cost_usd",
        ):
            if k in metrics:
                parts.append(f"- **{k}**: {metrics[k]}")
        parts.append("")
    parts.append("## Conversación")
    for turn in history:
        role = "Usuario" if turn["role"] == "user" else "Asistente"
        parts.append(f"### {role}")
        parts.append(turn["content"])
        parts.append("")
    return "\n".join(parts)


def conversation_to_pdf(markdown_text: str) -> bytes:
    """Convierte un markdown sencillo a PDF mínimo (sin estilos sofisticados).

    Usa ``reportlab`` si está disponible; si no, devuelve los bytes del MD como
    fallback para que la descarga no falle.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError:
        return markdown_text.encode("utf-8")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story: list[Any] = []
    for line in markdown_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        if line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Heading1"]))
        else:
            esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(esc, styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()
