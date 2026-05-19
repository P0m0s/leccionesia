"""
Comandos / del chat conversacional.

Cuando el ``transcript`` empieza por ``/`` se intercepta antes de llamar al LLM
y se ejecuta una acción local sobre la sesión. Reduce latencia, ahorra tokens
y da al usuario controles de "primera clase" sobre la conversación.

Comandos soportados:

- ``/reset`` (alias ``/clear``): vacía historial y metadata, conserva
  ``session_id``.
- ``/metadata``: muestra el ``project_metadata`` actual.
- ``/regenerate``: descarta la última respuesta del asistente y la regenera
  llamando al LLM (este caso es especial: NO se queda en este módulo, sino que
  delega de vuelta al orquestador).
- ``/help``: lista los comandos disponibles.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.sessions import ProjectMetadata, Session
from app.structured import StructuredEstimation


@dataclass
class CommandResult:
    text: str
    structured: StructuredEstimation
    structured_ok: bool = True
    needs_regenerate: bool = False
    """Si True, el router debe llamar al orquestador con el último user message."""


def is_command(transcript: str) -> bool:
    stripped = (transcript or "").strip()
    return stripped.startswith("/") and len(stripped) > 1


def _wrap(text: str) -> CommandResult:
    return CommandResult(
        text=text,
        structured=StructuredEstimation(summary_markdown=text),
    )


def _cmd_help() -> CommandResult:
    text = (
        "## Comandos disponibles\n"
        "- `/reset` (o `/clear`): vacía historial y metadata.\n"
        "- `/metadata`: muestra el `project_metadata` acumulado.\n"
        "- `/regenerate`: vuelve a generar la última respuesta del asistente.\n"
        "- `/help`: muestra esta ayuda."
    )
    return _wrap(text)


def _cmd_reset(session: Session) -> CommandResult:
    session.history._messages.clear()
    session.project_metadata = ProjectMetadata()
    return _wrap(
        "Sesión reiniciada. Historial vacío y `project_metadata` limpia.\n\n"
        "Continúa cuando quieras.",
    )


def _cmd_metadata(session: Session) -> CommandResult:
    pm = session.project_metadata
    if not any(
        (
            pm.project_name,
            pm.project_type,
            pm.assumed_team_size,
            pm.mentioned_technologies,
            pm.agreed_scope,
        ),
    ):
        return _wrap("Aún no hay `project_metadata` capturada en esta sesión.")
    lines = ["## project_metadata actual"]
    if pm.project_name:
        lines.append(f"- **project_name**: {pm.project_name}")
    if pm.project_type:
        lines.append(f"- **project_type**: {pm.project_type}")
    if pm.assumed_team_size is not None:
        lines.append(f"- **assumed_team_size**: {pm.assumed_team_size}")
    if pm.mentioned_technologies:
        lines.append(f"- **mentioned_technologies**: {', '.join(pm.mentioned_technologies)}")
    if pm.agreed_scope:
        lines.append(f"- **agreed_scope**: {pm.agreed_scope}")
    return _wrap("\n".join(lines))


def _cmd_regenerate(session: Session) -> CommandResult:
    """Quita la última respuesta del asistente y pide regeneración al orquestador."""
    msgs = session.history._messages
    if not msgs or msgs[-1].role != "assistant":
        return _wrap("No hay respuesta del asistente que regenerar.")

    msgs.pop()
    return CommandResult(
        text="",
        structured=StructuredEstimation(),
        structured_ok=False,
        needs_regenerate=True,
    )


def dispatch(transcript: str, session: Session) -> CommandResult | None:
    """Si ``transcript`` es un comando soportado, lo ejecuta y devuelve resultado.

    Si el transcript no es comando o no se reconoce, devuelve ``None``.
    """
    if not is_command(transcript):
        return None

    parts = transcript.strip().split(maxsplit=1)
    cmd = parts[0].lower()

    handlers = {
        "/help": lambda: _cmd_help(),
        "/reset": lambda: _cmd_reset(session),
        "/clear": lambda: _cmd_reset(session),
        "/metadata": lambda: _cmd_metadata(session),
        "/regenerate": lambda: _cmd_regenerate(session),
    }
    handler = handlers.get(cmd)
    if handler is None:
        return _wrap(
            f"Comando desconocido: `{cmd}`. Usa `/help` para ver los comandos disponibles.",
        )
    return handler()
