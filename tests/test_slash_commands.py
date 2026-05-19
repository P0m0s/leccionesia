"""Tests de los comandos / del chat conversacional."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.services.slash_commands import dispatch, is_command
from app.sessions import ProjectMetadata, Session


def test_is_command_detecta_correctamente() -> None:
    assert is_command("/help")
    assert is_command("/reset")
    assert is_command("  /metadata   ")
    assert not is_command("hola")
    assert not is_command("/")
    assert not is_command("")


def test_help_lista_comandos() -> None:
    session = Session(session_id="s1")
    result = dispatch("/help", session)
    assert result is not None
    assert "/reset" in result.text
    assert "/metadata" in result.text


def test_reset_vacia_historial_y_metadata() -> None:
    session = Session(session_id="s1")
    session.history.add_user_message("Aurora")
    session.history.add_assistant_message("ok")
    session.project_metadata = ProjectMetadata(project_name="Aurora")

    result = dispatch("/reset", session)
    assert result is not None
    assert session.history.messages == []
    assert session.project_metadata.project_name is None


def test_metadata_vacia_explica_estado() -> None:
    session = Session(session_id="s1")
    result = dispatch("/metadata", session)
    assert result is not None
    assert "Aún no hay" in result.text


def test_metadata_no_vacia_se_muestra() -> None:
    session = Session(session_id="s1")
    session.project_metadata = ProjectMetadata(
        project_name="Atlas",
        assumed_team_size=3,
        mentioned_technologies=["Python", "FastAPI"],
    )
    result = dispatch("/metadata", session)
    assert result is not None
    assert "Atlas" in result.text
    assert "Python" in result.text


def test_regenerate_sin_respuesta_previa_informa() -> None:
    session = Session(session_id="s1")
    session.history.add_user_message("Inicio")
    result = dispatch("/regenerate", session)
    assert result is not None
    assert result.needs_regenerate is False


def test_regenerate_quita_ultima_respuesta_y_pide_regeneracion() -> None:
    session = Session(session_id="s1")
    session.history.add_user_message("Inicio")
    session.history.add_assistant_message("Respuesta vieja")
    result = dispatch("/regenerate", session)
    assert result is not None
    assert result.needs_regenerate is True
    assert len(session.history.messages) == 1
    assert session.history.messages[-1].role == "user"


def test_comando_desconocido_devuelve_ayuda() -> None:
    session = Session(session_id="s1")
    result = dispatch("/foo", session)
    assert result is not None
    assert "desconocido" in result.text.lower()


@pytest.mark.asyncio
async def test_endpoint_slash_reset_no_llama_al_llm(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "El proyecto se llama Atlas con Python."},
    )
    calls_before = len(fake_llm["calls"])

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "/reset"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "reiniciada" in body["text"].lower()
    assert body["project_metadata"]["project_name"] is None
    assert len(fake_llm["calls"]) == calls_before


@pytest.mark.asyncio
async def test_endpoint_slash_regenerate_re_llama_al_llm(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    sid = create.json()["session_id"]

    await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "Quiero un MVP de marketplace."},
    )
    calls_before = len(fake_llm["calls"])

    resp = await async_client.post(
        f"/sessions/{sid}/estimate",
        data={"transcript": "/regenerate"},
    )
    assert resp.status_code == 200
    assert len(fake_llm["calls"]) == calls_before + 1
