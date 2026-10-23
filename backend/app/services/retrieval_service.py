
import json
import re
from pathlib import Path

from backend.app.services.embedding_service import embed_texts


def _normalize_text(text: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def search_documents(
    query: str,
    processed_dir: Path,
    top_k: int = 3,
    document_id: str | None = None,
) -> list[dict]:
    """Search document chunks using semantic similarity."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    # bool is a subclass of int in Python, so reject it explicitly.
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be at least 1 and must be an integer.")

    if document_id is not None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must not be blank.")
        document_id = document_id.strip()

    if not processed_dir.exists():
        return []

    documents = []
    seen_texts = set()

    for file_path in sorted(processed_dir.glob("*.json")):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                document = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(document, dict):
            continue

        chunks = document.get("chunks", [])
        if not isinstance(chunks, list):
            continue

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            if (
                document_id is not None
                and chunk.get("document_id") != document_id
            ):
                continue

            normalized_text = _normalize_text(text)
            if not normalized_text or normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)
            documents.append(chunk)

    if not documents:
        return []

    vectors = embed_texts(
        [query.strip()] + [chunk["text"] for chunk in documents]
    )

    if len(vectors) != len(documents) + 1:
        raise ValueError(
            "The embedding service returned an unexpected number of vectors."
        )

    query_vector = vectors[0]
    results = []

    for chunk, vector in zip(documents, vectors[1:]):
        if len(vector) != len(query_vector):
            raise ValueError("Embedding dimensions do not match.")

        # Normalized embeddings allow dot product to represent cosine similarity.
        score = sum(
            query_value * document_value
            for query_value, document_value in zip(query_vector, vector)
        )

        results.append(
            {
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "page_number": chunk.get("page_number"),
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk["text"],
                "score": float(score),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]
