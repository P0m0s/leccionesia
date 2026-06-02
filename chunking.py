"""Chunking por tokens (tiktoken) sin romper palabras."""

from __future__ import annotations

from typing import Any, TypedDict

import tiktoken

_ENCODING_NAME = "cl100k_base"


class ChunkRecord(TypedDict):
    chunk: str
    metadata: dict[str, Any]


def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def _count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def chunk_text(text: str, max_tokens: int = 300) -> list[str]:
    """Divide texto en chunks respetando límites de tokens y palabras completas."""
    stripped = text.strip()
    if not stripped:
        return []

    words = stripped.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = _count_tokens(word)
        separator_tokens = 1 if current_words else 0
        projected = current_tokens + separator_tokens + word_tokens

        if projected > max_tokens and current_words:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_tokens = word_tokens
        else:
            if current_words:
                current_tokens += separator_tokens
            current_words.append(word)
            current_tokens += word_tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_document(doc: str, max_tokens: int = 300) -> list[ChunkRecord]:
    """Wrapper de chunk_text con metadatos mínimos por chunk."""
    parts = chunk_text(doc, max_tokens=max_tokens)
    records: list[ChunkRecord] = []
    search_from = 0

    for chunk_id, chunk in enumerate(parts):
        start_char = doc.find(chunk, search_from)
        if start_char < 0:
            start_char = search_from
        end_char = start_char + len(chunk)
        records.append(
            {
                "chunk": chunk,
                "metadata": {
                    "chunk_id": chunk_id,
                    "start_char": start_char,
                    "end_char": end_char,
                },
            },
        )
        search_from = end_char

    return records
