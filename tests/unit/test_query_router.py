"""
Unit tests for QueryRouter module.
"""

from unittest.mock import MagicMock

from retrieval.query_router import QueryIntent, QueryRouter


def test_query_router_pattern_match_broad():
    router = QueryRouter()
    assert router.classify("Explain the entire document") == QueryIntent.BROAD
    assert router.classify("Summarize the pitch deck") == QueryIntent.BROAD
    assert router.classify("Give me an overview of the pdf") == QueryIntent.BROAD
    assert router.classify("What is this document about?") == QueryIntent.BROAD
    assert router.classify("Walk me through the main points") == QueryIntent.BROAD


def test_query_router_narrow_queries():
    router = QueryRouter()
    assert router.classify("What is the refund policy?") == QueryIntent.NARROW
    assert router.classify("Who is the CEO of the company?") == QueryIntent.NARROW
    assert router.classify("How much does tier 1 cost per month?") == QueryIntent.NARROW


def test_query_router_bm25_score_flatness_broad():
    router = QueryRouter(bm25_flatness_threshold=1.5)
    mock_bm25 = MagicMock()
    mock_bm25.corpus_size = 10
    # Flat score distribution: top_score = 1.2, top_5_mean = 1.0 -> ratio 1.2 < 1.5
    mock_bm25.get_scores.return_value = [1.2, 1.1, 1.0, 0.9, 0.8, 0.5, 0.2]

    intent = router.classify("some vague ambiguous query", bm25_retriever=mock_bm25)
    assert intent == QueryIntent.BROAD


def test_query_router_bm25_score_peaked_narrow():
    router = QueryRouter(bm25_flatness_threshold=1.5)
    mock_bm25 = MagicMock()
    mock_bm25.corpus_size = 10
    # Peaked score distribution: top_score = 12.0, top_5_mean = 3.0 -> ratio 4.0 > 1.5
    mock_bm25.get_scores.return_value = [12.0, 3.0, 1.0, 0.5, 0.1, 0.0, 0.0]

    intent = router.classify("specific keyword query", bm25_retriever=mock_bm25)
    assert intent == QueryIntent.NARROW
