"""Vector store en memoria con similitud coseno y persistencia JSON opcional."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embeddings_service import cosine_similarity


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add(self, embedding: list[float], metadata: dict[str, Any]) -> None:
        self._items.append({"embedding": embedding, "metadata": metadata})

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[tuple[float, dict[str, Any]]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._items:
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append((score, item["metadata"]))
        scored.sort(key=lambda row: row[0], reverse=True)
        return scored[:k]

    def size(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items.clear()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": self._items}
        target.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        target = Path(path)
        if not target.exists():
            return
        payload = json.loads(target.read_text(encoding="utf-8"))
        self._items = list(payload.get("items", []))


vector_store = InMemoryVectorStore()
