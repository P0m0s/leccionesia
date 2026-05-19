"""
Gestión de sesiones conversacionales en memoria.

Contiene:

- `ProjectMetadata`: contexto enriquecido del proyecto que se acumula turno a turno
  y se inyecta en el system prompt vía Jinja2.
- `ConversationMessage` / `ConversationHistory`: historial con ventana deslizante.
- `Session`: une historial y metadata por `session_id`.
- `SessionStore`: diccionario global en memoria (sin BBDD ni Redis).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Literal

from pydantic import BaseModel, Field

from app.prompts.loader import render_chat_system_prompt

MAX_TURNS = 6
"""Número máximo de mensajes (user/assistant combinados) que se conservan en la ventana."""

Role = Literal["user", "assistant", "system"]


class ProjectMetadata(BaseModel):
    """Contexto enriquecido del proyecto, acumulado a lo largo de la conversación."""

    project_name: str | None = None
    project_type: str | None = Field(
        default=None,
        description=(
            "Categoría heurística: mobile_app, web_saas, internal_tool, "
            "data_pipeline. Selecciona el bloque few-shot del system prompt."
        ),
    )
    assumed_team_size: int | None = None
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None
    conversation_summary: str | None = Field(
        default=None,
        description=(
            "Resumen acumulado de turnos antiguos que ya salieron de la "
            "ventana deslizante. Se inyecta en el system prompt para no perder "
            "contexto histórico."
        ),
    )

    def merged_with(self, other: ProjectMetadata) -> ProjectMetadata:
        """Combina la metadata actual con un parche, sin perder lo previamente acordado."""
        techs = list(self.mentioned_technologies)
        for tech in other.mentioned_technologies:
            if tech and tech not in techs:
                techs.append(tech)

        return ProjectMetadata(
            project_name=other.project_name or self.project_name,
            project_type=other.project_type or self.project_type,
            assumed_team_size=other.assumed_team_size
            if other.assumed_team_size is not None
            else self.assumed_team_size,
            mentioned_technologies=techs,
            agreed_scope=other.agreed_scope or self.agreed_scope,
            conversation_summary=other.conversation_summary or self.conversation_summary,
        )


class ConversationMessage(BaseModel):
    role: Role
    content: str


class ConversationHistory:
    """
    Lista limitada de mensajes (`user` y `assistant`) con ventana deslizante.

    El `system` prompt **no** se almacena aquí: se regenera en cada llamada vía
    `to_messages_list(project_metadata)` con Jinja2 para que refleje siempre el
    estado más reciente del `ProjectMetadata`.
    """

    def __init__(self, *, max_turns: int = MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._messages: deque[ConversationMessage] = deque()

    def _append(self, msg: ConversationMessage) -> ConversationMessage | None:
        """Añade ``msg`` aplicando el overflow manualmente.

        Devuelve el mensaje desplazado si la ventana se desbordó, o ``None``.
        """
        dropped: ConversationMessage | None = None
        if len(self._messages) >= self.max_turns:
            dropped = self._messages.popleft()
        self._messages.append(msg)
        return dropped

    def add_user_message(self, content: str) -> ConversationMessage | None:
        return self._append(ConversationMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> ConversationMessage | None:
        return self._append(ConversationMessage(role="assistant", content=content))

    @property
    def messages(self) -> list[ConversationMessage]:
        return list(self._messages)

    def to_messages_list(
        self,
        project_metadata: ProjectMetadata,
        *,
        prompt_version: str = "v1",
    ) -> list[dict[str, str]]:
        """
        Devuelve la lista lista-para-LLM: ``[system, ...últimos N mensajes]``.

        El system prompt se regenera con Jinja2 inyectando ``project_metadata``.
        """
        system_prompt = render_chat_system_prompt(project_metadata, version=prompt_version)
        out: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in self._messages:
            out.append({"role": msg.role, "content": msg.content})
        return out


class SessionMetrics(BaseModel):
    """Acumulado de métricas y costes por sesión."""

    turns_count: int = 0
    llm_calls: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    elapsed_ms_total: float = 0.0
    attachments_processed_total: int = 0
    images_processed_total: int = 0
    estimated_cost_usd: float = 0.0
    last_model: str | None = None
    last_provider: str | None = None
    last_metrics: dict = Field(default_factory=dict)


class Session(BaseModel):
    """Sesión conversacional persistida en memoria."""

    session_id: str
    project_metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    created_at: float = Field(default_factory=time.time)
    last_activity_at: float = Field(default_factory=time.time)
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._history = ConversationHistory()

    @property
    def history(self) -> ConversationHistory:
        return self._history

    def touch(self) -> None:
        self.last_activity_at = time.time()


class SessionStore:
    """Diccionario global en memoria. Sin BBDD ni Redis (estado in-process)."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def reset(self) -> None:
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)

    def items(self):
        return self._sessions.items()

    def remove(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def evict_inactive(self, *, ttl_seconds: int, now: float | None = None) -> int:
        """Elimina sesiones cuya última actividad sea anterior a ``now - ttl``."""
        current = now if now is not None else time.time()
        cutoff = current - ttl_seconds
        stale = [sid for sid, s in self._sessions.items() if s.last_activity_at < cutoff]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)


session_store = SessionStore()
"""Singleton global usado por el router."""
