"""
Modelo de salida **estructurada** para las estimaciones conversacionales.

Se le pide al LLM que devuelva un único objeto JSON con esta forma para que el
cliente pueda:

- Sumar totales sin parsear texto.
- Renderizar tablas nativas en Streamlit.
- Exportar a Markdown / PDF de forma fiable.
- Comparar estimaciones entre turnos.

Si el LLM devuelve markdown libre o un JSON inválido, el orquestador hace
*fallback*: copia todo el texto a `summary_markdown` y deja vacíos el resto de
campos, sin romper el contrato HTTP.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

TShirt = Literal["XS", "S", "M", "L", "XL"]
Severity = Literal["low", "medium", "high"]


class Assumption(BaseModel):
    text: str


class Risk(BaseModel):
    text: str
    severity: Severity | None = None


class LineItem(BaseModel):
    """Una línea de trabajo dentro de la estimación."""

    name: str
    description: str | None = None
    area: str | None = Field(
        default=None,
        description="frontend, backend, datos, infra, qa, diseño, gestión…",
    )
    t_shirt: TShirt | None = None
    hours_min: float | None = Field(default=None, ge=0)
    hours_max: float | None = Field(default=None, ge=0)


class Phase(BaseModel):
    """Fase del proyecto si el output_format prefiere fases sobre line items."""

    name: str
    objective: str | None = None
    deliverables: list[str] = Field(default_factory=list)
    duration_weeks_min: float | None = Field(default=None, ge=0)
    duration_weeks_max: float | None = Field(default=None, ge=0)
    risks: list[str] = Field(default_factory=list)


class StructuredEstimation(BaseModel):
    """Respuesta canónica de un turno conversacional."""

    summary_markdown: str = Field(
        default="",
        description="Resumen narrativo. El cliente lo muestra siempre.",
    )
    line_items: list[LineItem] = Field(default_factory=list)
    phases: list[Phase] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    total_hours_min: float | None = Field(default=None, ge=0)
    total_hours_max: float | None = Field(default=None, ge=0)
    confidence: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Autoconfianza del modelo (0–10) en la estimación de este turno.",
    )
    next_step: str | None = Field(
        default=None,
        description="Próximo paso sugerido por el modelo.",
    )

    def computed_totals(self) -> tuple[float | None, float | None]:
        """Si el modelo no rellenó los totales, calcúlalos desde los line items."""
        if self.total_hours_min is not None and self.total_hours_max is not None:
            return self.total_hours_min, self.total_hours_max

        min_total = sum(li.hours_min or 0 for li in self.line_items) or None
        max_total = sum(li.hours_max or 0 for li in self.line_items) or None
        return min_total, max_total


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)

_SMART_QUOTES = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u2013": "-",
        "\u2014": "-",
    },
)


def _normalize_quotes(text: str) -> str:
    """Reemplaza comillas tipográficas por ASCII para que ``json.loads`` no falle."""
    return text.translate(_SMART_QUOTES) if text else text


def _extract_json_payload(text: str) -> str | None:
    """Intenta extraer un objeto JSON top-level desde la respuesta del LLM."""
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]

    return None


def _try_parse_json(payload: str) -> dict | None:
    """Intenta ``json.loads`` con varios saneados progresivos."""
    candidates = [payload, _normalize_quotes(payload)]
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def parse_structured_response(text: str) -> tuple[StructuredEstimation, bool]:
    """
    Devuelve ``(estimación, parseado_ok)``.

    Si el LLM no devuelve JSON válido, vuelve un ``StructuredEstimation`` con
    ``summary_markdown=text`` y ``parseado_ok=False`` para que el cliente sepa
    que está en modo degradado.
    """
    payload = _extract_json_payload(text)
    if payload is None:
        return StructuredEstimation(summary_markdown=text.strip()), False

    data = _try_parse_json(payload)
    if data is None:
        return StructuredEstimation(summary_markdown=text.strip()), False

    try:
        est = StructuredEstimation.model_validate(data)
    except ValidationError:
        return StructuredEstimation(summary_markdown=text.strip()), False

    # Defensa: si el LLM (incorrectamente) metió el JSON entero como string
    # dentro de ``summary_markdown``, intentamos extraer el campo interno y
    # quedarnos solo con el texto narrativo.
    sm = (est.summary_markdown or "").strip()
    if sm.startswith("{") and sm.endswith("}"):
        inner = _try_parse_json(sm)
        if isinstance(inner, dict) and isinstance(inner.get("summary_markdown"), str):
            est = est.model_copy(update={"summary_markdown": inner["summary_markdown"]})

    if not est.summary_markdown:
        # Fallback: nunca devolvemos JSON crudo al usuario. Generamos un texto
        # mínimo a partir de los campos que tengamos.
        est = est.model_copy(
            update={
                "summary_markdown": _fallback_summary(est),
            },
        )

    if est.total_hours_min is None or est.total_hours_max is None:
        tmin, tmax = est.computed_totals()
        est = est.model_copy(
            update={
                "total_hours_min": est.total_hours_min if est.total_hours_min is not None else tmin,
                "total_hours_max": est.total_hours_max if est.total_hours_max is not None else tmax,
            },
        )

    return est, True


def _fallback_summary(est: StructuredEstimation) -> str:
    """Genera un summary mínimo cuando el LLM olvidó ``summary_markdown``."""
    parts: list[str] = ["## Estimación preliminar"]
    if est.line_items:
        parts.append(f"Detectados {len(est.line_items)} ítems de trabajo.")
    if est.next_step:
        parts.append(f"**Próximo paso:** {est.next_step}")
    return "\n\n".join(parts) if len(parts) > 1 else "Estimación recibida sin resumen narrativo."


STRUCTURED_OUTPUT_INSTRUCTIONS = """\
## Formato de salida OBLIGATORIO

Responde **siempre** con un único objeto JSON, sin texto fuera del JSON ni *code fences*. El JSON debe ajustarse a este schema:

```
{
  "summary_markdown": "string (resumen en markdown que el usuario verá)",
  "line_items": [
    {
      "name": "string",
      "description": "string | null",
      "area": "frontend | backend | datos | infra | qa | diseño | gestión | otro",
      "t_shirt": "XS | S | M | L | XL | null",
      "hours_min": number | null,
      "hours_max": number | null
    }
  ],
  "phases": [
    {
      "name": "string",
      "objective": "string | null",
      "deliverables": ["string"],
      "duration_weeks_min": number | null,
      "duration_weeks_max": number | null,
      "risks": ["string"]
    }
  ],
  "assumptions": [{"text": "string"}],
  "risks": [{"text": "string", "severity": "low | medium | high | null"}],
  "total_hours_min": number | null,
  "total_hours_max": number | null,
  "confidence": number entero 0..10,
  "next_step": "string | null"
}
```

### Reglas JSON (críticas)
- Usa **solamente comillas dobles ASCII** (`"`). **Nunca** uses comillas tipográficas/curvas (`"`, `"`, `'`, `'`).
- **No** envuelvas el JSON en *code fences* (```json ... ```). Solo el objeto JSON crudo.
- **No** repitas el JSON dentro de `summary_markdown`. `summary_markdown` es texto narrativo en markdown, NO contiene JSON.
- Escapa correctamente saltos de línea y comillas internas.

### Reglas de contenido
- Usa **rangos** en lugar de números puntuales (`hours_min`/`hours_max`). Respeta la tabla T-shirt.
- **Da valor desde el turno 1**. Aunque el usuario aporte poca info, propón una **primera estimación tentativa** y lista lo que **asumes** para llegar a ella. No te quedes solo preguntando.
- Marca como `assumptions` cualquier dato no aportado por el usuario.
- `confidence`: 0 = especulativo, 10 = muy seguro. Con 1 turno de poco contexto usa 3–4. Con scope claro y stack acordado, 6–8.
- `next_step` debe ser **una sola pregunta o acción concreta** para el siguiente turno (máx. 2 líneas).

### Estilo del `summary_markdown`
- Empieza con un encabezado breve (`## Estimación preliminar` o `## Plan propuesto`).
- Si hay ítems estimados, **describe el alcance en 1-2 frases** (no listes los items, eso ya va en `line_items`).
- Si estás aclarando, usa esta plantilla:
  ```
  ## Primera lectura

  <1-2 frases con tu interpretación inicial y un rango de horas tentativo>

  **Para afinar necesito saber:**
  - <pregunta 1>
  - <pregunta 2>
  - <pregunta 3>
  ```
- Tono profesional, directo y conciso. Sin frases de relleno ni disculpas.
- **Máximo 6 líneas** en `summary_markdown`. El detalle va en `line_items`, `assumptions`, `risks`.
"""
