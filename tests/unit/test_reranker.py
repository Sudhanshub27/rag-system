from unittest.mock import MagicMock

import pytest

from retrieval.reranker import CrossEncoderReranker
from utils.models import Chunk, RetrievedChunk


@pytest.fixture
def mock_cross_encoder(mocker):
    mock_model = MagicMock()
    # Return dummy scores for pairs
    mock_model.predict.return_value = [0.2, 0.95]
    mocker.patch("sentence_transformers.CrossEncoder", return_value=mock_model)
    return mock_model


def test_reranker_empty_input(mock_cross_encoder):
    reranker = CrossEncoderReranker(model_name="dummy-model", top_n=2)
    assert reranker.rerank("query", []) == []


def test_reranker_success(mock_cross_encoder):
    reranker = CrossEncoderReranker(model_name="dummy-model", top_n=2)

    c1 = Chunk(text="Lower relevant text", source="a.txt", chunk_id="c1")
    c2 = Chunk(text="Higher relevant text", source="b.txt", chunk_id="c2")

    rc1 = RetrievedChunk(chunk=c1, score=0.5)
    rc2 = RetrievedChunk(chunk=c2, score=0.4)

    reranked = reranker.rerank("query", [rc1, rc2])

    assert len(reranked) == 2
    assert reranked[0].chunk.chunk_id == "c2"
    assert reranked[0].score == 0.95
    assert reranked[0].rank == 1
    assert reranked[1].rank == 2


def test_reranker_exception_fallback(mocker):
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Prediction error")
    mocker.patch("sentence_transformers.CrossEncoder", return_value=mock_model)

    reranker = CrossEncoderReranker(model_name="dummy-model", top_n=1)

    c1 = Chunk(text="Text 1", source="a.txt", chunk_id="c1")
    c2 = Chunk(text="Text 2", source="b.txt", chunk_id="c2")
    rc1 = RetrievedChunk(chunk=c1, score=0.5)
    rc2 = RetrievedChunk(chunk=c2, score=0.4)

    # Should gracefully degrade to original top_n candidates
    fallback = reranker.rerank("query", [rc1, rc2])
    assert len(fallback) == 1
    assert fallback[0].chunk.chunk_id == "c1"
