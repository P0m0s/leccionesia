"""Orquestación: chunking + embeddings + vector store."""

from __future__ import annotations

from embeddings_service import EmbeddingService
from chunking import chunk_document
from vector_store import InMemoryVectorStore


def index_text(
    text: str,
    store: InMemoryVectorStore,
    embedder: EmbeddingService,
    *,
    max_tokens: int = 300,
) -> int:
    """Chunkea, embeddea y guarda en el store. Devuelve número de chunks indexados."""
    records = chunk_document(text, max_tokens=max_tokens)
    if not records:
        return 0

    embeddings = embedder.embed_many([record["chunk"] for record in records])
    for embedding, record in zip(embeddings, records, strict=True):
        metadata = {**record["metadata"], "chunk": record["chunk"]}
        store.add(embedding, metadata)

    return len(records)
