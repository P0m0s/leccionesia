"""Test 1: ``project_metadata`` se actualiza entre turnos de la misma sesión."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_project_metadata_se_actualiza_entre_turnos(
    async_client: AsyncClient,
    fake_llm: dict,
) -> None:
    create = await async_client.post("/sessions")
    assert create.status_code == 201
    session_id = create.json()["session_id"]
    assert session_id

    turn_1 = await async_client.post(
        f"/sessions/{session_id}/estimate",
        data={
            "transcript": (
                "Hola, el proyecto se llama Aurora. Vamos a usar Python y FastAPI."
            ),
        },
    )
    assert turn_1.status_code == 200, turn_1.text
    metadata_1 = turn_1.json()["project_metadata"]

    assert metadata_1["project_name"] == "Aurora"
    assert "Python" in metadata_1["mentioned_technologies"]
    assert "FastAPI" in metadata_1["mentioned_technologies"]
    assert metadata_1["assumed_team_size"] is None

    turn_2 = await async_client.post(
        f"/sessions/{session_id}/estimate",
        data={
            "transcript": (
                "Confirmo que tendremos un equipo de 4 personas y añadimos React "
                "al stack. Alcance: MVP en 6 semanas."
            ),
        },
    )
    assert turn_2.status_code == 200, turn_2.text
    metadata_2 = turn_2.json()["project_metadata"]

    assert metadata_2["project_name"] == "Aurora"
    assert metadata_2["assumed_team_size"] == 4
    assert "Python" in metadata_2["mentioned_technologies"]
    assert "FastAPI" in metadata_2["mentioned_technologies"]
    assert "React" in metadata_2["mentioned_technologies"]
    assert metadata_2["agreed_scope"]
    assert "MVP" in metadata_2["agreed_scope"]

    assert metadata_2 != metadata_1
    assert len(fake_llm["calls"]) == 2
