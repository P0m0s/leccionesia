from fastapi import APIRouter, HTTPException

from app.schemas import (
    EmbedRequest,
    EmbedResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from embeddings_service import EmbeddingService
from indexing import index_text
from vector_store import vector_store

router = APIRouter(tags=["embeddings"])


def _embedder() -> EmbeddingService:
    try:
        return EmbeddingService()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/embed", response_model=EmbedResponse)
def embed_document(body: EmbedRequest) -> EmbedResponse:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El campo text no puede estar vacío.")

    embedder = _embedder()
    chunks_indexed = index_text(text, vector_store, embedder)
    return EmbedResponse(
        chunks_indexed=chunks_indexed,
        store_size=vector_store.size(),
    )


@router.post("/search", response_model=SearchResponse)
def search_documents(body: SearchRequest) -> SearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="El campo query no puede estar vacío.")

    if vector_store.size() == 0:
        return SearchResponse(results=[])

    embedder = _embedder()
    query_embedding = embedder.embed_text(query)
    hits = vector_store.search(query_embedding, k=body.k)

    results = [
        SearchResultItem(
            score=score,
            chunk=metadata.get("chunk", ""),
            metadata={k: v for k, v in metadata.items() if k != "chunk"},
        )
        for score, metadata in hits
    ]
    return SearchResponse(results=results)
