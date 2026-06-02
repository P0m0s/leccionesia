"""Tests del pipeline embeddings + chunking + vector store."""

from __future__ import annotations

import pytest

from chunking import chunk_document, chunk_text
from embeddings_service import EmbeddingService, cosine_similarity
from indexing import index_text
from vector_store import InMemoryVectorStore


def test_chunking_long_text_multiple_chunks() -> None:
    words = ["palabra"] * 400
    text = " ".join(words)
    chunks = chunk_text(text, max_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        for word in chunk.split():
            assert word == "palabra"


def test_chunking_respects_max_tokens() -> None:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = " ".join(f"token{i}" for i in range(200))
    max_tokens = 30
    chunks = chunk_text(text, max_tokens=max_tokens)
    for chunk in chunks:
        assert len(enc.encode(chunk)) <= max_tokens


def test_chunking_does_not_break_words() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk_text(text, max_tokens=8)
    for chunk in chunks:
        for token in chunk.split():
            assert token in text


def test_chunk_document_metadata() -> None:
    doc = "Primera frase larga. Segunda frase con más contenido para trocear."
    records = chunk_document(doc, max_tokens=10)
    assert records
    for record in records:
        meta = record["metadata"]
        assert "chunk_id" in meta
        assert "start_char" in meta
        assert "end_char" in meta
        assert doc[meta["start_char"] : meta["end_char"]] == record["chunk"]


def test_embeddings_equal_texts_similar() -> None:
    service = EmbeddingService(backend="local")
    a = service.embed_text("mismo contenido semántico")
    b = service.embed_text("mismo contenido semántico")
    assert cosine_similarity(a, b) > 0.99


def test_embeddings_different_texts_dissimilar() -> None:
    service = EmbeddingService(backend="local")
    a = service.embed_text("estimación de APIs REST con FastAPI")
    b = service.embed_text("receta de tarta de manzana con canela")
    assert cosine_similarity(a, b) < cosine_similarity(a, a) * 0.95


def test_search_top1_relevant_chunk() -> None:
    store = InMemoryVectorStore()
    embedder = EmbeddingService(backend="local")
    chunks = [
        "El proyecto usa Python y FastAPI para el backend.",
        "La base de datos principal es PostgreSQL con migraciones Alembic.",
        "El frontend es una app móvil en React Native con notificaciones push.",
    ]
    for text in chunks:
        index_text(text, store, embedder, max_tokens=200)

    query_embedding = embedder.embed_text("API REST en Python con FastAPI")
    results = store.search(query_embedding, k=3)
    assert results
    top_score, top_meta = results[0]
    assert "FastAPI" in top_meta["chunk"]
    assert top_score >= results[1][0]


@pytest.mark.asyncio
async def test_embed_and_search_endpoints(async_client) -> None:
    embed_resp = await async_client.post("/embed", json={"text": "Vector store con cosine similarity."})
    assert embed_resp.status_code == 200
    body = embed_resp.json()
    assert body["chunks_indexed"] >= 1
    assert body["store_size"] >= 1

    search_resp = await async_client.post(
        "/search",
        json={"query": "similitud coseno vectorial", "k": 3},
    )
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert results
    assert "score" in results[0]
    assert "chunk" in results[0]
    assert "metadata" in results[0]
