
import json
import math
from pathlib import Path

import faiss
import numpy as np

from backend.app.services.embedding_service import embed_texts


INDEX_FILENAME = "banking_controls.faiss"
METADATA_FILENAME = "banking_controls_metadata.json"


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be at least 1 and must be an integer.")


def _validate_document_id(document_id: str | None) -> str | None:
    if document_id is not None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must not be blank.")
        return document_id.strip()
    return None


def _metadata_for_chunk(chunk: dict) -> dict:
    text = chunk.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Every chunk must contain non-empty text.")

    document_id = chunk.get("document_id")
    filename = chunk.get("filename")
    page_number = chunk.get("page_number")
    chunk_index = chunk.get("chunk_index")

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Every chunk must contain a valid document_id.")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Every chunk must contain a valid filename.")

    if (
        not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or page_number < 1
    ):
        raise ValueError("Every chunk must contain a positive page_number.")

    if (
        not isinstance(chunk_index, int)
        or isinstance(chunk_index, bool)
        or chunk_index < 0
    ):
        raise ValueError("Every chunk must contain a non-negative chunk_index.")

    return {
        "document_id": document_id,
        "filename": filename,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text": text.strip(),
    }


def save_vector_index(chunks: list[dict], index_dir: Path) -> int:
    """Build and persist an exact FAISS inner-product index."""

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("At least one chunk is required to build the index.")

    metadata = [_metadata_for_chunk(chunk) for chunk in chunks]
    vectors = embed_texts([item["text"] for item in metadata])

    if len(vectors) != len(metadata):
        raise ValueError("The embedding service returned an unexpected number of vectors.")

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
        raise ValueError("The embedding service returned invalid vector dimensions.")

    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("The embedding service returned invalid vector values.")

    norms = np.linalg.norm(matrix, axis=1)

    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero vectors.")

    # Inner-product search on normalized vectors gives cosine similarity.
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / INDEX_FILENAME
    metadata_path = index_dir / METADATA_FILENAME
    temporary_index_path = index_dir / f"{INDEX_FILENAME}.tmp"
    temporary_metadata_path = index_dir / f"{METADATA_FILENAME}.tmp"

    try:
        faiss.write_index(index, str(temporary_index_path))

        with temporary_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.flush()

        # Replace the previous files only after both temporary files are ready.
        temporary_index_path.replace(index_path)
        temporary_metadata_path.replace(metadata_path)
    finally:
        temporary_index_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)

    return index.ntotal


def load_vector_index(index_dir: Path) -> tuple[faiss.Index, list[dict]]:
    """Load the persisted FAISS index and its matching metadata."""

    index_dir = Path(index_dir)
    index_path = index_dir / INDEX_FILENAME
    metadata_path = index_dir / METADATA_FILENAME

    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("The vector index or its metadata has not been created.")

    try:
        index = faiss.read_index(str(index_path))

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        raise ValueError("The saved vector index could not be loaded.") from exc

    if not isinstance(metadata, list):
        raise ValueError("The saved vector metadata has an invalid format.")

    if index.ntotal != len(metadata):
        raise ValueError("The vector index and metadata are out of sync.")

    if index.ntotal > 0 and index.d <= 0:
        raise ValueError("The saved vector index has invalid dimensions.")

    return index, metadata


def _word_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    """Return normalized consecutive-word shingles for overlap comparison."""
    words = " ".join(text.casefold().split()).split()

    if not words:
        return set()

    if len(words) < size:
        return {tuple(words)}

    return {
        tuple(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def _is_near_duplicate(
    candidate_text: str,
    accepted_text: str,
    threshold: float = 0.80,
) -> bool:
    """Detect when most of the shorter passage is already represented."""
    candidate_shingles = _word_shingles(candidate_text)
    accepted_shingles = _word_shingles(accepted_text)

    if not candidate_shingles or not accepted_shingles:
        return False

    shorter_count = min(len(candidate_shingles), len(accepted_shingles))
    overlap_count = len(candidate_shingles & accepted_shingles)

    return overlap_count / shorter_count >= threshold


def search_vector_index(
    query: str,
    index_dir: Path,
    top_k: int = 3,
    document_id: str | None = None,
) -> list[dict]:
    """Search saved vectors and optionally restrict results to one document."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    _validate_top_k(top_k)
    document_id = _validate_document_id(document_id)

    index, metadata = load_vector_index(index_dir)

    if index.ntotal == 0:
        return []

    query_vectors = embed_texts([query.strip()])

    if len(query_vectors) != 1:
        raise ValueError("The embedding service returned an unexpected number of vectors.")

    query_vector = np.asarray(query_vectors, dtype=np.float32)

    if query_vector.ndim != 2 or query_vector.shape != (1, index.d):
        raise ValueError("Query embedding dimensions do not match the saved index.")

    if not np.isfinite(query_vector).all():
        raise ValueError("The query embedding contains invalid values.")

    if math.isclose(float(np.linalg.norm(query_vector)), 0.0):
        raise ValueError("The query embedding must not be a zero vector.")

    faiss.normalize_L2(query_vector)

    # Examine every stored vector before filtering so document-specific
    # searches don't miss matching passages outside a global top_k.
    scores, positions = index.search(query_vector, index.ntotal)

    results = []
    seen_texts = set()

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        chunk = metadata[int(position)]

        if document_id is not None and chunk["document_id"] != document_id:
            continue

        normalized_text = " ".join(chunk["text"].casefold().split())

        if normalized_text in seen_texts:
            continue

        if any(
            _is_near_duplicate(chunk["text"], result["text"])
            for result in results
        ):
            continue

        seen_texts.add(normalized_text)

        results.append(
            {
                **chunk,
                "score": float(score),
            }
        )

        if len(results) >= top_k:
            break

    return results

def add_chunks_to_vector_index(chunks: list[dict], index_dir: Path) -> int:
    """Add new document chunks to the existing persistent FAISS index."""

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("At least one chunk is required to update the index.")

    index_dir = Path(index_dir)
    index_path = index_dir / INDEX_FILENAME
    metadata_path = index_dir / METADATA_FILENAME

    index_exists = index_path.is_file()
    metadata_exists = metadata_path.is_file()

    if index_exists != metadata_exists:
        raise ValueError(
            "The vector index and metadata are out of sync. "
            "Rebuild the index from processed documents."
        )

    new_metadata = [_metadata_for_chunk(chunk) for chunk in chunks]

    if index_exists:
        index, existing_metadata = load_vector_index(index_dir)
    else:
        index = None
        existing_metadata = []

    existing_keys = {
        (
            item["document_id"],
            item["page_number"],
            item["chunk_index"],
        )
        for item in existing_metadata
    }

    unique_metadata = []
    for item in new_metadata:
        key = (
            item["document_id"],
            item["page_number"],
            item["chunk_index"],
        )
        if key not in existing_keys:
            existing_keys.add(key)
            unique_metadata.append(item)

    if not unique_metadata:
        return len(existing_metadata)

    vectors = embed_texts([item["text"] for item in unique_metadata])
    matrix = np.asarray(vectors, dtype=np.float32)

    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(unique_metadata)
        or matrix.shape[1] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("The embedding service returned invalid vectors.")

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero vectors.")

    faiss.normalize_L2(matrix)

    if index is None:
        index = faiss.IndexFlatIP(matrix.shape[1])
    elif index.d != matrix.shape[1]:
        raise ValueError("Embedding dimensions do not match the saved index.")

    if not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("The saved index uses an unsupported FAISS index type.")

    index.add(matrix)
    combined_metadata = existing_metadata + unique_metadata

    index_dir.mkdir(parents=True, exist_ok=True)
    temporary_index_path = index_dir / f"{INDEX_FILENAME}.tmp"
    temporary_metadata_path = index_dir / f"{METADATA_FILENAME}.tmp"

    try:
        faiss.write_index(index, str(temporary_index_path))

        with temporary_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(combined_metadata, file, ensure_ascii=False, indent=2)
            file.flush()

        # Publish the new index and metadata after both temporary files are written.
        temporary_index_path.replace(index_path)
        temporary_metadata_path.replace(metadata_path)
    finally:
        temporary_index_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)

    return len(combined_metadata)

def test_incremental_indexing_adds_new_document_without_losing_existing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)

    initial_chunks = sample_chunks()[:1]
    assert vector_store.add_chunks_to_vector_index(initial_chunks, tmp_path) == 1

    new_chunk = [
        {
            "document_id": "doc-new",
            "filename": "new-controls.pdf",
            "page_number": 5,
            "chunk_index": 0,
            "text": "Credit risk monitoring requirements.",
        }
    ]

    assert vector_store.add_chunks_to_vector_index(new_chunk, tmp_path) == 2

    index, metadata = vector_store.load_vector_index(tmp_path)

    assert index.ntotal == 2
    assert {item["document_id"] for item in metadata} == {"doc-a", "doc-new"}


def test_incremental_indexing_does_not_duplicate_same_chunk(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)

    chunks = sample_chunks()[:1]

    assert vector_store.add_chunks_to_vector_index(chunks, tmp_path) == 1
    assert vector_store.add_chunks_to_vector_index(chunks, tmp_path) == 1

    index, metadata = vector_store.load_vector_index(tmp_path)

    assert index.ntotal == 1
    assert len(metadata) == 1
