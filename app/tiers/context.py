"""Identidad del *caller* y selección de tier.

El backend nunca confía en un campo `tier` que venga "suelto" en el body.
La identidad (incluido el tier) llega siempre como un `CallerContext`
construido por una dependency de FastAPI.

Tenemos dos estrategias canónicas:

- **Opción A — JWT firmado** (recomendado a medio plazo). El servicio de
  identidad emite tokens HS256 que incluyen `user_id` y `tier` como claims.
  Mantenemos el esqueleto comentado abajo para activarlo cuando exista un
  servicio de auth real.
- **Opción B — Headers simples en red privada** (activa). En este MVP el
  frontend Streamlit corre sin auth, así que aceptamos un header
  `X-Estimator-Tier`. Como esto solo es seguro detrás de un perímetro de
  confianza, lo marcamos explícitamente como tal.

Si no llega ningún tier en el header, devolvemos `None` y el endpoint sigue
funcionando con el flujo conversacional clásico (compat hacia atrás).
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import Header, HTTPException
from pydantic import BaseModel

TierName = Literal["developer", "pm", "executive", "research"]
_VALID_TIERS: frozenset[str] = frozenset({"developer", "pm", "executive", "research"})


class CallerContext(BaseModel):
    """Identidad mínima de quien hace la petición."""

    user_id: str
    tier: TierName


async def get_caller_context(
    x_estimator_user: str | None = Header(default=None),
    x_estimator_tier: str | None = Header(default=None),
) -> CallerContext | None:
    """Construye el ``CallerContext`` desde headers (Opción B).

    Devuelve ``None`` si el cliente no envía ningún tier — en ese caso el
    endpoint cae al flujo conversacional clásico, sin patrón tier.
    """
    if not x_estimator_tier:
        return None

    if x_estimator_tier not in _VALID_TIERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tier desconocido: {x_estimator_tier!r}. "
                f"Válidos: {sorted(_VALID_TIERS)}"
            ),
        )

    return CallerContext(
        user_id=x_estimator_user or "anon",
        tier=x_estimator_tier,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Opción A — JWT firmado (DESACTIVADA, plantilla para producción).
#
# Para activarla:
#  1. `pip install python-jose[cryptography]` (añadir a pyproject.toml).
#  2. Definir `ESTIMATOR_JWT_SECRET` en el entorno (HS256) o
#     `ESTIMATOR_JWT_PUBLIC_KEY` (RS256).
#  3. Sustituir `get_caller_context` por `get_caller_context_jwt` en el router.
#
# from fastapi import Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from jose import JWTError, jwt
#
# _JWT_SECRET = os.environ.get("ESTIMATOR_JWT_SECRET", "")
# _JWT_ALG = "HS256"
# _bearer = HTTPBearer(auto_error=False)
#
# async def get_caller_context_jwt(
#     creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
# ) -> CallerContext:
#     if creds is None:
#         raise HTTPException(status_code=401, detail="Falta token bearer")
#     try:
#         payload = jwt.decode(creds.credentials, _JWT_SECRET, algorithms=[_JWT_ALG])
#     except JWTError as e:
#         raise HTTPException(status_code=401, detail=f"JWT inválido: {e}") from e
#     tier = payload.get("tier")
#     if tier not in _VALID_TIERS:
#         raise HTTPException(status_code=403, detail=f"Tier desconocido: {tier!r}")
#     return CallerContext(user_id=payload["sub"], tier=tier)
# ---------------------------------------------------------------------------


__all__ = ["CallerContext", "TierName", "get_caller_context"]
# Mantenemos `os` referenciado para que la rama A comentada compile si la
# alguien la descomenta sin tocar imports.
_ = os
