from unittest.mock import MagicMock

import pytest

from retrieval.hybrid_retriever import HybridRetriever
from utils.models import Chunk, RetrievedChunk


@pytest.fixture
def retriever():
    vector_store = MagicMock()
    embed_fn = MagicMock()
    return HybridRetriever(
        vector_store=vector_store,
        embed_fn=embed_fn,
        bm25_weight=0.5,
        vector_weight=0.5,
    )


def test_rrf_empty_inputs(retriever):
    result = retriever._reciprocal_rank_fusion([], [])
    assert result == []


def test_rrf_both_ranks_higher_than_single_rank(retriever):
    chunk_a = Chunk(text="Doc A text", source="a.txt", chunk_id="chunk_a")
    chunk_b = Chunk(text="Doc B text", source="b.txt", chunk_id="chunk_b")

    bm25_results = [
        RetrievedChunk(chunk=chunk_a, score=10.0),
        RetrievedChunk(chunk=chunk_b, score=5.0),
    ]
    vector_results = [
        RetrievedChunk(chunk=chunk_a, score=0.9),
    ]

    fused = retriever._reciprocal_rank_fusion(bm25_results, vector_results, k=60)

    assert len(fused) == 2
    assert fused[0].chunk.chunk_id == "chunk_a"
    assert fused[1].chunk.chunk_id == "chunk_b"
    assert fused[0].score > fused[1].score


def test_rrf_exact_formula_hand_computed(retriever):
    chunk_a = Chunk(text="Doc A text", source="a.txt", chunk_id="chunk_a")

    # rank 1 in BM25
    bm25_results = [RetrievedChunk(chunk=chunk_a, score=10.0)]
    # rank 2 in Vector (put a dummy at rank 1)
    dummy_chunk = Chunk(text="Dummy", source="d.txt", chunk_id="dummy")
    vector_results = [
        RetrievedChunk(chunk=dummy_chunk, score=0.95),
        RetrievedChunk(chunk=chunk_a, score=0.85),
    ]

    # Formula:
    # BM25 rank 1 contribution: 0.5 / (60 + 1) = 0.5 / 61
    # Vector rank 2 contribution: 0.5 / (60 + 2) = 0.5 / 62
    expected_score = (0.5 / 61.0) + (0.5 / 62.0)

    fused = retriever._reciprocal_rank_fusion(bm25_results, vector_results, k=60)
    chunk_a_res = next(r for r in fused if r.chunk.chunk_id == "chunk_a")

    assert chunk_a_res.score == pytest.approx(expected_score, rel=1e-5)


def test_rrf_deduplication_by_chunk_id(retriever):
    chunk_a = Chunk(text="Duplicate chunk text", source="a.txt", chunk_id="chunk_dup")

    bm25_results = [RetrievedChunk(chunk=chunk_a, score=12.0)]
    vector_results = [RetrievedChunk(chunk=chunk_a, score=0.99)]

    fused = retriever._reciprocal_rank_fusion(bm25_results, vector_results, k=60)

    assert len(fused) == 1
    assert fused[0].chunk.chunk_id == "chunk_dup"
