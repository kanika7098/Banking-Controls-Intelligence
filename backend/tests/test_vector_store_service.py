
import json

import pytest

from backend.app.services import vector_store_service as vector_store


def fake_embeddings(texts):
    vectors = []

    for text in texts:
        lowered = text.casefold()

        if "credit" in lowered:
            vectors.append([0.0, 1.0, 0.0])
        elif "customer" in lowered:
            vectors.append([0.0, 0.0, 1.0])
        else:
            vectors.append([1.0, 0.0, 0.0])

    return vectors


def sample_chunks():
    return [
        {
            "document_id": "doc-a",
            "filename": "controls.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Banking risk controls and monitoring.",
        },
        {
            "document_id": "doc-b",
            "filename": "credit-policy.pdf",
            "page_number": 4,
            "chunk_index": 0,
            "text": "Credit risk assessment and policy.",
        },
        {
            "document_id": "doc-c",
            "filename": "customer-policy.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "text": "Customer protection and complaints.",
        },
    ]


def test_save_and_reload_persistent_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)

    count = vector_store.save_vector_index(sample_chunks(), tmp_path)

    assert count == 3
    assert (tmp_path / vector_store.INDEX_FILENAME).is_file()
    assert (tmp_path / vector_store.METADATA_FILENAME).is_file()

    index, metadata = vector_store.load_vector_index(tmp_path)

    assert index.ntotal == 3
    assert len(metadata) == 3
    assert metadata[1]["filename"] == "credit-policy.pdf"


def test_search_returns_most_similar_chunk_first(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)
    vector_store.save_vector_index(sample_chunks(), tmp_path)

    results = vector_store.search_vector_index(
        query="credit risk",
        index_dir=tmp_path,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["document_id"] == "doc-b"
    assert results[0]["page_number"] == 4
    assert isinstance(results[0]["score"], float)


def test_search_can_filter_by_document_id(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)
    vector_store.save_vector_index(sample_chunks(), tmp_path)

    results = vector_store.search_vector_index(
        query="credit risk",
        index_dir=tmp_path,
        top_k=3,
        document_id="doc-a",
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-a"


def test_search_rejects_blank_query(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)
    vector_store.save_vector_index(sample_chunks(), tmp_path)

    with pytest.raises(ValueError, match="Query must not be empty"):
        vector_store.search_vector_index("   ", tmp_path)


def test_search_rejects_invalid_top_k(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)
    vector_store.save_vector_index(sample_chunks(), tmp_path)

    with pytest.raises(ValueError, match="top_k"):
        vector_store.search_vector_index("credit", tmp_path, top_k=0)


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="has not been created"):
        vector_store.load_vector_index(tmp_path)


def test_invalid_chunk_is_rejected(tmp_path):
    invalid_chunks = [
        {
            "document_id": "doc-a",
            "filename": "controls.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "   ",
        }
    ]

    with pytest.raises(ValueError, match="non-empty text"):
        vector_store.save_vector_index(invalid_chunks, tmp_path)


def test_metadata_count_mismatch_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)
    vector_store.save_vector_index(sample_chunks(), tmp_path)

    metadata_path = tmp_path / vector_store.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop()

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="out of sync"):
        vector_store.load_vector_index(tmp_path)

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



def test_search_deduplicates_identical_text(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)

    chunks = [
        {
            "document_id": "doc-a",
            "filename": "controls.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Security controls require continuous monitoring.",
        },
        {
            "document_id": "doc-a",
            "filename": "controls.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "text": "  SECURITY CONTROLS require continuous monitoring.  ",
        },
        {
            "document_id": "doc-b",
            "filename": "other-controls.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "text": "Customer protection and complaints.",
        },
    ]

    vector_store.save_vector_index(chunks, tmp_path)

    results = vector_store.search_vector_index(
        query="banking controls",
        index_dir=tmp_path,
        top_k=3,
    )

    assert len(results) == 2
    assert len({
        " ".join(result["text"].casefold().split())
        for result in results
    }) == 2

def test_search_filters_substantially_overlapping_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "embed_texts", fake_embeddings)

    shared_text = (
        "Banks must monitor security incidents continuously and escalate "
        "confirmed incidents to the security operations center."
    )

    chunks = [
        {
            "document_id": "doc-a",
            "filename": "controls.pdf",
            "page_number": 20,
            "chunk_index": 0,
            "text": shared_text,
        },
        {
            "document_id": "doc-a",
            "filename": "controls.pdf",
            "page_number": 20,
            "chunk_index": 1,
            "text": (
                "Additional context. "
                + shared_text
                + " Teams must document the response."
            ),
        },
        {
            "document_id": "doc-b",
            "filename": "other-controls.pdf",
            "page_number": 4,
            "chunk_index": 0,
            "text": "Customer complaints must be tracked and resolved promptly.",
        },
    ]

    vector_store.save_vector_index(chunks, tmp_path)

    results = vector_store.search_vector_index(
        query="security incident monitoring",
        index_dir=tmp_path,
        top_k=3,
    )

    normalized_results = [
        " ".join(result["text"].casefold().split())
        for result in results
    ]

    assert len(normalized_results) == len(set(normalized_results))
    assert len(results) == 2
