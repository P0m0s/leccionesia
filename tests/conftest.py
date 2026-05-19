"""Fixtures comunes a los tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.services.attachments import reset_cache
from app.sessions import session_store


@pytest.fixture(autouse=True)
def _reset_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cada test arranca con el store de sesiones vacío y caché limpio.

    También se desactivan por defecto los flags que disparan llamadas LLM extra
    (summary automático y extractor LLM de metadata). Los tests que los
    necesiten los activan explícitamente con sus propias fixtures.
    """
    session_store.reset()
    reset_cache()
    monkeypatch.setattr(settings, "enable_auto_summary", False)
    monkeypatch.setattr(settings, "enable_llm_metadata", False)


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Sustituye `generate_chat_messages` por un fake determinista.

    El fake guarda en su `state["calls"]` la lista de mensajes recibidos en cada
    llamada. Por defecto devuelve markdown plano (el orquestador hace fallback a
    ``structured_ok=False``). Si se asigna ``state["response_builder"]`` a una
    función ``(messages) -> str``, ese builder se usa en su lugar.
    """
    state: dict[str, Any] = {"calls": [], "response_builder": None}

    def _default_response(messages: list[dict[str, str]]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        has_attachment = "--- attachment:" in last_user
        n_chars = len(last_user)
        return (
            "Estimación simulada (fake LLM).\n"
            f"- tamaño_entrada_chars: {n_chars}\n"
            f"- con_adjunto: {has_attachment}\n"
            "Próximo paso sugerido: definir alcance."
        )

    def _fake(messages, *, images=None, metrics_out=None):
        state["calls"].append(list(messages))
        state.setdefault("images_per_call", []).append(list(images or []))
        builder = state["response_builder"] or _default_response
        text = builder(messages)
        if metrics_out is not None:
            metrics_out.update(
                {
                    "model": "gpt-4o-mini",
                    "provider": "fake",
                    "input_tokens": 100,
                    "output_tokens": len(text) // 4,
                    "elapsed_ms": 1.0,
                },
            )
        return text, "gpt-4o-mini", "fake"

    monkeypatch.setattr(
        "app.services.session_service.generate_chat_messages",
        _fake,
    )
    monkeypatch.setattr(
        "app.tiers.pipelines.generate_chat_messages",
        _fake,
    )
    return state


@pytest.fixture
def fake_llm_stream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Sustituye ``stream_chat_messages`` por un fake que emite chunks de un texto.

    ``state["text"]`` es lo que devolverá troceado en N chunks; por defecto un
    JSON estructurado mínimo.
    """
    state: dict[str, Any] = {
        "calls": [],
        "text": (
            '{"summary_markdown": "## OK", '
            '"line_items": [], "phases": [], "assumptions": [], "risks": [], '
            '"confidence": 5, "next_step": "siguiente"}'
        ),
        "chunk_size": 8,
    }

    def _fake(messages, *, metrics_out=None, images=None):
        state["calls"].append(list(messages))
        state.setdefault("images_per_call", []).append(list(images or []))
        text = state["text"]
        chunk = state["chunk_size"]
        for i in range(0, len(text), chunk):
            yield text[i : i + chunk]
        if metrics_out is not None:
            metrics_out.update(
                {
                    "model": "fake-stream-model",
                    "provider": "fake",
                    "input_tokens": 10,
                    "output_tokens": len(text) // 4,
                    "elapsed_ms": 1.0,
                },
            )

    monkeypatch.setattr(
        "app.services.session_service.stream_chat_messages",
        _fake,
    )
    return state


@pytest_asyncio.fixture
async def async_client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:  # pragma: no cover - infra
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
