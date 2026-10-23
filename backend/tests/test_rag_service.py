
import pytest

from backend.app.services import rag_service


@pytest.fixture
def sample_results():
    return [
        {
            "text": "Banks should monitor security logs and investigate incidents.",
            "document_id": "doc-123",
            "filename": "rbi.pdf",
            "page_number": 7,
            "chunk_index": 2,
            "score": 0.85,
        }
    ]


def test_answer_question_uses_retrieved_evidence(
    monkeypatch, sample_results, tmp_path
):
    monkeypatch.setattr(
        rag_service,
        "search_vector_index",
        lambda **kwargs: sample_results,
    )

    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "Banks should monitor security logs. [Source 1]"

    monkeypatch.setattr(
        rag_service, "_generate_with_ollama", fake_generate
    )

    result = rag_service.answer_question(
        query="What should banks monitor?",
        index_dir=tmp_path,
    )

    assert "monitor security logs" in result["answer"]
    assert result["grounded"] is True
    assert result["retrieved_chunks"] == 1
    assert result["citations"][0]["page_number"] == 7
    assert "[Source 1]" in captured["prompt"]
    assert sample_results[0]["text"] in captured["prompt"]


def test_answer_question_returns_insufficient_evidence_when_no_results(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        rag_service,
        "search_vector_index",
        lambda **kwargs: [],
    )

    def unexpected_generation(prompt):
        pytest.fail("LLM should not be called when no evidence is retrieved")

    monkeypatch.setattr(
        rag_service, "_generate_with_ollama", unexpected_generation
    )

    result = rag_service.answer_question(
        query="What is the bank's incident response policy?",
        index_dir=tmp_path,
    )

    assert result["grounded"] is False
    assert result["citations"] == []
    assert result["retrieved_chunks"] == 0
    assert "could not find relevant evidence" in result["answer"].lower()

def test_answer_question_returns_insufficient_evidence_below_threshold(
    monkeypatch, sample_results, tmp_path
):
    low_score_results = [
        {**sample_results[0], "score": 0.05}
    ]

    monkeypatch.setattr(
        rag_service,
        "search_vector_index",
        lambda **kwargs: low_score_results,
    )

    monkeypatch.setattr(
        rag_service,
        "_generate_with_ollama",
        lambda prompt: pytest.fail(
            "LLM should not be called for low-relevance evidence"
        ),
    )

    result = rag_service.answer_question(
        query="What should banks monitor?",
        index_dir=tmp_path,
    )

    assert result["grounded"] is False
    assert result["citations"] == []
    assert result["retrieved_chunks"] == 1


def test_answer_question_propagates_ollama_failure(
    monkeypatch, sample_results, tmp_path
):
    monkeypatch.setattr(
        rag_service,
        "search_vector_index",
        lambda **kwargs: sample_results,
    )

    def failed_generation(prompt):
        raise RuntimeError("Could not connect to Ollama.")

    monkeypatch.setattr(
        rag_service, "_generate_with_ollama", failed_generation
    )

    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        rag_service.answer_question(
            query="What should banks monitor?",
            index_dir=tmp_path,
        )


def test_answer_question_rejects_empty_query(tmp_path):
    with pytest.raises(ValueError, match="Query must not be empty"):
        rag_service.answer_question(
            query="   ",
            index_dir=tmp_path,
        )

def test_answer_question_handles_answer_without_source_labels(
    monkeypatch, sample_results, tmp_path
):
    monkeypatch.setattr(
        rag_service,
        "search_vector_index",
        lambda **kwargs: sample_results,
    )

    monkeypatch.setattr(
        rag_service,
        "_generate_with_ollama",
        lambda prompt: (
            "Banks should monitor security logs and investigate incidents."
        ),
    )

    result = rag_service.answer_question(
        query="What should banks monitor?",
        index_dir=tmp_path,
    )

    assert result["answer"]
    assert result["citations"] == []
    assert result["grounded"] is False
