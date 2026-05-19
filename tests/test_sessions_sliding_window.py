"""Test 3: la ventana deslizante nunca supera ``MAX_TURNS`` mensajes."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.sessions import MAX_TURNS, ConversationHistory, ProjectMetadata, session_store


def test_to_messages_list_respeta_max_turns_unit() -> None:
    history = ConversationHistory()
    metadata = ProjectMetadata()

    for i in range(8):
        history.add_user_message(f"pregunta {i}")
        history.add_assistant_message(f"respuesta {i}")

    rendered = history.to_messages_list(metadata)
    non_system = [m for m in rendered if m["role"] != "system"]
    assert len(non_system) <= MAX_TURNS
    assert len(history.messages) <= MAX_TURNS

    contents = [m["content"] for m in non_system]
    assert any("respuesta 7" in c for c in contents)
    assert all("pregunta 0" not in c for c in contents)


@pytest.mark.asyncio
async def test_sesion_de_8_turnos_no_excede_max_turns(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    for i in range(8):
        resp = await async_client.post(
            f"/sessions/{sid}/estimate",
            data={"transcript": f"Turno {i}: refinar alcance del proyecto."},
        )
        assert resp.status_code == 200, resp.text

    last_call_messages = fake_llm["calls"][-1]
    non_system_in_last_call = [m for m in last_call_messages if m["role"] != "system"]
    assert len(non_system_in_last_call) <= MAX_TURNS

    session = session_store.get(sid)
    assert session is not None
    assert len(session.history.messages) <= MAX_TURNS

    rendered = session.history.to_messages_list(session.project_metadata)
    assert sum(1 for m in rendered if m["role"] == "system") == 1

    contents = [m["content"] for m in rendered if m["role"] != "system"]
    assert all("Turno 0" not in c for c in contents)
