"""Patrón **tier** para perfiles de cliente.

Cada `tier` (developer / pm / executive / research) selecciona:

- un **template Jinja2** específico (instrucciones y vocabulario distintos),
- un **schema Pydantic** distinto (estructuras de salida diferentes),
- una **pipeline** distinta (`single_call` síncrono vs `deep_research`).

Anti-patrones evitados (siguiendo la lección AI Engineering 2026/04 — *Prompts
adaptativos por perfil de usuario: el patrón tier*):

1. **Tier desde el frontend sin verificación**: el backend siempre exige un
   `CallerContext` vía `Depends(get_caller_context)`. Aunque el frontend del
   MVP envíe el tier en un header, la dependency es el único punto de entrada
   y en producción se sustituye por JWT/cookies/etc. sin tocar el resto del
   código.
2. **Schema único con branching**: cada tier tiene su propio Pydantic model.
3. **Templates divergentes sin includes**: usamos parciales compartidos
   (`_project_metadata.j2`, `_reference_estimates.j2`).
4. **Cambiar solo el tono**: cada tier cambia *estructura* + *vocabulario* +
   (en research) pipeline completa.
"""

from app.tiers.config import (
    PIPELINE_HANDLERS,
    TIER_CONFIG,
    TierName,
    resolve_tier_config,
)
from app.tiers.context import CallerContext, get_caller_context
from app.tiers.schemas import (
    DeveloperEstimate,
    ExecutiveEstimate,
    PmEstimate,
    ResearchEstimate,
)

__all__ = [
    "PIPELINE_HANDLERS",
    "TIER_CONFIG",
    "CallerContext",
    "DeveloperEstimate",
    "ExecutiveEstimate",
    "PmEstimate",
    "ResearchEstimate",
    "TierName",
    "get_caller_context",
    "resolve_tier_config",
]
