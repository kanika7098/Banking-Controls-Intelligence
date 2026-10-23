

import json

import pytest

from backend.app.services import retrieval_service
from backend.app.services.retrieval_service import search_documents


def fake_embeddings(texts):
    """Return deterministic vectors without loading the real model."""
    return [[1.0, 0.0] for _ in texts]


def write_document(directory, filename, document):
    path = directory / filename
    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    return path


def test_empty_query_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Query must not be empty"):
        search_documents(
            query="   ",
            processed_dir=tmp_path,
        )


def test_empty_directory_returns_no_results(tmp_path):
    results = search_documents(
        query="banking controls",
        processed_dir=tmp_path,
    )

    assert results == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(tmp_path, top_k):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        search_documents(
            query="banking controls",
            processed_dir=tmp_path,
            top_k=top_k,
        )


def test_document_id_filter_returns_only_matching_document(
    tmp_path, monkeypatch
):
    documents = [
        {
            "chunks": [
                {
                    "document_id": "doc-a",
                    "filename": "a.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "text": "Banking risk controls and monitoring.",
                }
            ]
        },
        {
            "chunks": [
                {
                    "document_id": "doc-b",
                    "filename": "b.pdf",
                    "page_number": 2,
                    "chunk_index": 0,
                    "text": "Credit risk and regulatory compliance.",
                }
            ]
        },
    ]

    write_document(tmp_path, "a.json", documents[0])
    write_document(tmp_path, "b.json", documents[1])

    monkeypatch.setattr(
        retrieval_service,
        "embed_texts",
        fake_embeddings,
    )

    results = search_documents(
        query="banking controls",
        processed_dir=tmp_path,
        document_id="doc-b",
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-b"
    assert results[0]["filename"] == "b.pdf"


def test_duplicate_text_is_returned_only_once(tmp_path, monkeypatch):
    chunks = [
        {
            "document_id": "doc-a",
            "filename": "a.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Banking risk controls and monitoring.",
        },
        {
            "document_id": "doc-b",
            "filename": "b.pdf",
            "page_number": 3,
            "chunk_index": 5,
            "text": "  BANKING risk controls   and monitoring. ",
        },
    ]

    write_document(
        tmp_path,
        "a.json",
        {"chunks": [chunks[0]]},
    )
    write_document(
        tmp_path,
        "b.json",
        {"chunks": [chunks[1]]},
    )

    monkeypatch.setattr(
        retrieval_service,
        "embed_texts",
        fake_embeddings,
    )

    results = search_documents(
        query="banking risk controls",
        processed_dir=tmp_path,
        top_k=10,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-a"
    assert results[0]["text"] == chunks[0]["text"]
