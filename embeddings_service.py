"""Servicio de embeddings (OpenAI o local determinista para tests/dev)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal

from openai import OpenAI

from app.config import settings

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_DIM = 64
_LOCAL_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "con",
        "de",
        "del",
        "el",
        "en",
        "es",
        "la",
        "las",
        "lo",
        "los",
        "para",
        "por",
        "un",
        "una",
        "y",
    },
)


def _tokenize_for_local(text: str) -> list[str]:
    return re.findall(r"[a-z0-9áéíóúñü]+", text.strip().lower())


def _local_embed(text: str, *, dim: int = LOCAL_EMBEDDING_DIM) -> list[float]:
    """Embedding determinista sin API (útil en tests y sin clave)."""
    vec = [0.0] * dim
    tokens = _tokenize_for_local(text)
    for token in tokens:
        if token in _LOCAL_STOPWORDS:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            vec[i % dim] += (byte - 128) / 128.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class EmbeddingService:
    """Genera embeddings con OpenAI o backend local."""

    def __init__(
        self,
        *,
        backend: Literal["openai", "local"] | None = None,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._backend = backend or settings.embedding_backend
        self._model = model or settings.embedding_model
        self._batch_size = batch_size or settings.embedding_batch_size
        self._client: OpenAI | None = None
        if self._backend == "openai":
            key = api_key or settings.openai_api_key
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY requerida para embeddings OpenAI. "
                    "Usa EMBEDDING_BACKEND=local para modo sin API.",
                )
            self._client = OpenAI(api_key=key)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._backend == "local":
            return [_local_embed(t) for t in texts]

        assert self._client is not None
        results: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            ordered = sorted(response.data, key=lambda row: row.index)
            results.extend(row.embedding for row in ordered)
        return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
