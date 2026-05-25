"""
Genera PDFs determinísticos de tamaños aproximados: 5, 20, 50, 100 KB.

Uso:
    uv run python -m evals.stress.fixtures.build_pdfs

Los PDFs se escriben en ``evals/stress/fixtures/`` y NO deben commitearse
(añadir al .gitignore).  Solo se commitea este script.
"""

from __future__ import annotations

import pathlib
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

_HERE = pathlib.Path(__file__).resolve().parent

_FILLER = (
    "Este documento contiene especificaciones tecnicas de ejemplo para un "
    "proyecto de software ficticio. Incluye requisitos funcionales, requisitos "
    "no funcionales, restricciones de arquitectura, estimaciones de esfuerzo "
    "y criterios de aceptacion. El contenido es repetitivo a proposito para "
    "alcanzar el tamano objetivo del archivo PDF de forma deterministica. "
    "Requisito RF-{idx:04d}: el sistema debe procesar la solicitud en menos "
    "de 200ms bajo carga normal de 500 usuarios concurrentes y almacenar "
    "los resultados en una base de datos relacional con respaldo automatico."
)

_TARGET_SIZES_KB = [5, 20, 50, 100]


def build_pdf(target_kb: int, output_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Genera un PDF de al menos ``target_kb`` KB."""
    dest = (output_dir or _HERE) / f"synthetic_{target_kb}kb.pdf"
    target_bytes = target_kb * 1024

    multiplier = 4
    while True:
        _write_pdf(dest, target_kb, n_blocks=max(1, (target_bytes * multiplier) // len(_FILLER)))
        actual = dest.stat().st_size
        if actual >= target_bytes * 0.8:
            break
        multiplier = int(multiplier * (target_bytes / max(actual, 1)) * 1.3)

    return dest


def _write_pdf(dest: pathlib.Path, target_kb: int, n_blocks: int) -> None:
    c = canvas.Canvas(str(dest), pagesize=A4)
    width, height = A4
    margin = 50
    line_height = 10
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Documento Sintetico - {target_kb} KB")
    y -= 25
    c.setFont("Helvetica", 8)

    for i in range(n_blocks):
        text = _FILLER.format(idx=i + 1)
        for line in textwrap.wrap(text, width=105):
            if y < 40:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 8)
            c.drawString(margin, y, line)
            y -= line_height

    c.save()


def main() -> None:
    _HERE.mkdir(parents=True, exist_ok=True)
    for kb in _TARGET_SIZES_KB:
        path = build_pdf(kb)
        actual_kb = path.stat().st_size / 1024
        print(f"  OK {path.name}: {actual_kb:.1f} KB")


if __name__ == "__main__":
    main()
