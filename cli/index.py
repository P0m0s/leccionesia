"""Indexa un archivo de texto en el vector store."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from embeddings_service import EmbeddingService
from indexing import index_text
from vector_store import vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa un documento de texto.")
    parser.add_argument("--file", required=True, help="Ruta al archivo .txt (u otro texto plano).")
    parser.add_argument(
        "--persist",
        default=None,
        help="Ruta JSON opcional para persistir el vector store tras indexar.",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"No existe el archivo: {path}")

    text = path.read_text(encoding="utf-8")
    embedder = EmbeddingService()
    chunks = index_text(text, vector_store, embedder)

    persist_path = args.persist or settings.vector_store_path
    if persist_path:
        vector_store.save(persist_path)

    print(f"chunks_indexed={chunks} store_size={vector_store.size()}")


if __name__ == "__main__":
    main()
