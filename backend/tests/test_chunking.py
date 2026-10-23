import pytest

from backend.app.services.chunking_service import chunk_document_pages


def test_short_page_creates_one_chunk():
    pages = [{"page_number": 1, "text": "Banking controls reduce risk."}]

    chunks = chunk_document_pages(
        pages,
        document_id="doc-001",
        filename="controls.pdf",
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Banking controls reduce risk."
    assert chunks[0].document_id == "doc-001"
    assert chunks[0].filename == "controls.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_long_page_creates_overlapping_chunks():
    pages = [{"page_number": 1, "text": "A" * 2500}]

    chunks = chunk_document_pages(
        pages,
        document_id="doc-001",
        filename="controls.pdf",
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert len(chunks) == 3
    assert len(chunks[0].text) == 1000
    assert chunks[0].text[850:] == chunks[1].text[:150]
    assert chunks[1].text[850:] == chunks[2].text[:150]


def test_chunks_prefer_word_boundaries():
    text = ("Banking security controls require continuous monitoring. " * 40)
    pages = [{"page_number": 1, "text": text}]

    chunks = chunk_document_pages(
        pages,
        document_id="doc-001",
        filename="controls.pdf",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    for chunk in chunks[:-1]:
        assert not chunk.text.endswith(" ")
        assert len(chunk.text) <= 100


def test_chunk_indexes_are_sequential_across_pages():
    pages = [
        {"page_number": 1, "text": "First page content."},
        {"page_number": 2, "text": "Second page content."},
    ]

    chunks = chunk_document_pages(
        pages,
        document_id="doc-001",
        filename="controls.pdf",
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.page_number for chunk in chunks] == [1, 2]


def test_empty_pages_are_skipped():
    pages = [
        {"page_number": 1, "text": ""},
        {"page_number": 2, "text": "   "},
    ]

    chunks = chunk_document_pages(
        pages,
        document_id="doc-001",
        filename="empty.pdf",
    )

    assert chunks == []


def test_invalid_chunk_settings_are_rejected():
    pages = [{"page_number": 1, "text": "Example text."}]

    with pytest.raises(ValueError):
        chunk_document_pages(
            pages,
            document_id="doc-001",
            filename="controls.pdf",
            chunk_size=100,
            chunk_overlap=100,
        )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (100, -1)],
)
def test_invalid_chunk_size_or_overlap_is_rejected(
    chunk_size,
    chunk_overlap,
):
    with pytest.raises(ValueError):
        chunk_document_pages(
            [{"page_number": 1, "text": "Example text."}],
            document_id="doc-001",
            filename="controls.pdf",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
