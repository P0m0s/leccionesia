"""Tests de las utilidades de frontend (tarjetas, tablas, exportación)."""

from __future__ import annotations

from app.frontend_utils import (
    conversation_to_markdown,
    conversation_to_pdf,
    metadata_card_specs,
    parse_markdown_tables,
)


def test_metadata_card_specs_completa() -> None:
    metadata = {
        "project_name": "Atlas",
        "project_type": "web_saas",
        "assumed_team_size": 3,
        "mentioned_technologies": ["Python", "FastAPI"],
        "agreed_scope": "MVP en 6 semanas",
    }
    cards = metadata_card_specs(metadata)
    labels = {c["label"] for c in cards}
    assert {"Proyecto", "Tipo", "Equipo", "Tecnologías", "Alcance"}.issubset(labels)
    tipo_card = next(c for c in cards if c["label"] == "Tipo")
    assert "🌐" in tipo_card["icon"]


def test_metadata_card_specs_vacia_devuelve_lista_vacia() -> None:
    assert metadata_card_specs({}) == []
    assert metadata_card_specs({"project_name": "", "mentioned_technologies": []}) == []


def test_parse_markdown_tables_extrae_filas() -> None:
    text = """
Algo de prosa.

| Talla | Horas |
|-------|-------|
| XS    | 4-8   |
| S     | 16-40 |

Más texto.
"""
    tables = parse_markdown_tables(text)
    assert len(tables) == 1
    table = tables[0]
    assert table[0] == ["Talla", "Horas"]
    assert table[1] == ["XS", "4-8"]
    assert table[2] == ["S", "16-40"]


def test_parse_markdown_tables_sin_tablas() -> None:
    assert parse_markdown_tables("solo texto plano") == []


def test_conversation_to_markdown_contiene_secciones() -> None:
    md = conversation_to_markdown(
        session_id="abc",
        history=[
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Hola, ¿en qué te ayudo?"},
        ],
        metadata={"project_name": "Atlas", "mentioned_technologies": ["Python"]},
        metrics={"turns_count": 1, "llm_calls": 1, "input_tokens_total": 100, "output_tokens_total": 50, "estimated_cost_usd": 0.001},
    )
    assert "## project_metadata" in md
    assert "## Métricas" in md
    assert "## Conversación" in md
    assert "Hola" in md
    assert "Python" in md


def test_conversation_to_pdf_devuelve_bytes() -> None:
    md = "# Título\n\nTexto"
    pdf_bytes = conversation_to_pdf(md)
    assert isinstance(pdf_bytes, bytes) and len(pdf_bytes) > 100
    assert pdf_bytes.startswith(b"%PDF")
