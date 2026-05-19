"""
Tabla simple de costes por modelo, expresados en **USD por 1 K tokens**.

Las cifras son aproximaciones publicadas por los proveedores en distintos
momentos; sirven como **estimación**, no como facturación real. Cuando un
modelo no esté en la tabla, ``estimate_cost_usd`` devuelve ``None``.
"""

from __future__ import annotations

_PRICES_USD_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-opus-20240229": (0.015, 0.075),
}


def estimate_cost_usd(
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if not model or input_tokens is None or output_tokens is None:
        return None
    base = model.split(":", 1)[0]
    price = _PRICES_USD_PER_1K.get(base)
    if price is None:
        return None
    in_cost = (input_tokens / 1000.0) * price[0]
    out_cost = (output_tokens / 1000.0) * price[1]
    return in_cost + out_cost
