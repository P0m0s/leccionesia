"""Orquestación: plantillas Jinja, caché exact-match y llamadas al wrapper LLM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any, Literal

import structlog

from app.prompts.loader import render_estimation_prompt
from app.schemas import EstimationRequest, EstimationResponse
from app.services.llm_service import (
    generate_estimation_messages,
    stream_estimation_messages,
)

logger = structlog.get_logger(__name__)

PromptVersion = Literal["v1", "v2"]

_EXACT_CACHE: dict[str, str] = {}


def _cache_key(request: EstimationRequest, prompt_version: str) -> str:
    payload = {"prompt_version": prompt_version, **request.model_dump(mode="json")}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _log_rendered_prompt(
    *,
    prompt_version: str,
    system_prompt: str,
    user_prompt: str,
) -> None:
    logger.info(
        "estimation_prompt_rendered",
        prompt_version=prompt_version,
        system_chars=len(system_prompt),
        user_chars=len(user_prompt),
    )


def run_estimate_sync(
    request: EstimationRequest,
    prompt_version: PromptVersion,
) -> EstimationResponse:
    key = _cache_key(request, prompt_version)
    if key in _EXACT_CACHE:
        logger.debug(
            "estimation_cache_hit",
            prompt_version=prompt_version,
            cache_key_prefix=key[:16],
        )
        return EstimationResponse(text=_EXACT_CACHE[key], prompt_version=prompt_version)

    system_prompt, user_prompt = render_estimation_prompt(request, version=prompt_version)
    _log_rendered_prompt(
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    text, _model, _provider = generate_estimation_messages(system_prompt, user_prompt)
    _EXACT_CACHE[key] = text
    return EstimationResponse(text=text, prompt_version=prompt_version)


def iter_estimate_stream(
    request: EstimationRequest,
    prompt_version: PromptVersion,
    *,
    metrics_out: dict[str, Any] | None = None,
) -> Iterator[str]:
    key = _cache_key(request, prompt_version)
    if key in _EXACT_CACHE:
        yield _EXACT_CACHE[key]
        return

    system_prompt, user_prompt = render_estimation_prompt(request, version=prompt_version)
    _log_rendered_prompt(
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    parts: list[str] = []
    for chunk in stream_estimation_messages(
        system_prompt,
        user_prompt,
        metrics_out=metrics_out,
    ):
        parts.append(chunk)
        yield chunk

    if parts:
        _EXACT_CACHE[key] = "".join(parts)
