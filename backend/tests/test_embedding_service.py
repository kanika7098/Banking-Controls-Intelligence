
from unittest.mock import patch

import pytest

from backend.app.services.embedding_service import embed_texts


def test_empty_input_returns_empty_list():
    assert embed_texts([]) == []


def test_blank_text_is_rejected():
    with pytest.raises(ValueError):
        embed_texts(["   "])


def test_embeddings_are_generated():
    fake_model = type(
        "FakeModel",
        (),
        {
            "encode": lambda self, texts, **kwargs: [
                [0.6, 0.8] for _ in texts
            ],
        },
    )()

    with patch(
        "backend.app.services.embedding_service.get_embedding_model",
        return_value=fake_model,
    ):
        result = embed_texts(["Banking controls", "Risk management"])

    assert len(result) == 2
    assert result[0] == [0.6, 0.8]
    assert result[1] == [0.6, 0.8]
