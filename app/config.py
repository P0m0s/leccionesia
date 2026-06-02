from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    llm_provider: Literal["openai", "anthropic"] = Field(default="openai")
    openai_model: str = Field(default="gpt-4o-mini")
    anthropic_model: str = Field(default="claude-3-5-haiku-20241022")

    enable_llm_metadata: bool = Field(
        default=False,
        description=(
            "Si True, tras cada turno se hace una segunda llamada LLM ligera "
            "para enriquecer ProjectMetadata. Activarlo gasta tokens extra."
        ),
    )
    enable_auto_summary: bool = Field(
        default=True,
        description=(
            "Si True, al desbordar la ventana deslizante se llama al LLM para "
            "generar un resumen comprimido que se incorpora a "
            "ProjectMetadata.conversation_summary."
        ),
    )
    session_ttl_seconds: int = Field(
        default=24 * 3600,
        description="Tiempo sin actividad tras el cual una sesión se descarta.",
    )

    embedding_backend: Literal["openai", "local"] = Field(
        default="openai",
        description="Proveedor de embeddings: OpenAI API o local determinista (tests).",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Modelo OpenAI para embeddings cuando embedding_backend=openai.",
    )
    embedding_batch_size: int = Field(
        default=100,
        description="Tamaño de lote para embed_many con OpenAI.",
    )
    vector_store_path: str | None = Field(
        default=None,
        description="Ruta JSON opcional para persistir/cargar el vector store en disco.",
    )


settings = Settings()
