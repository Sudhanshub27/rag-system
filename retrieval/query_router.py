"""
Query Router
Classifies user queries into NARROW (specific chunk retrieval) or BROAD (document-level summary) intents.

Routing strategy:
1. Pattern match (regex/keyword pass, free): matches broad intent patterns like explain/summarize/overview/gist.
2. BM25 score-shape fallback (free): analyzes BM25 score distribution. Flat scores across chunks indicate a broad query.
3. Model classification fallback (cheap LLM call, optional): last resort for ambiguous queries.
"""

import re
from enum import Enum
from typing import Any

from utils.logger import logger


class QueryIntent(str, Enum):
    NARROW = "NARROW"
    BROAD = "BROAD"


# Patterns that strongly indicate a broad, document-level query
BROAD_PATTERN = re.compile(
    r"\b("
    r"explain|summarize|summarise|overview|walk\s+me\s+through|gist|"
    r"entire|whole|all|pitch\s+deck|document\s+summary|pdf\s+summary|"
    r"what\s+is\s+this\s+document\s+about|what\s+is\s+this\s+pdf\s+about|"
    r"key\s+takeaways|main\_points|table\s+of\s+contents|outline"
    r")\b",
    re.IGNORECASE,
)


class QueryRouter:
    """
    Classifies incoming user queries into NARROW or BROAD intents to select
    the optimal retrieval and generation path.
    """

    def __init__(
        self,
        broad_pattern: re.Pattern = BROAD_PATTERN,
        bm25_flatness_threshold: float = 1.5,
    ):
        self.broad_pattern = broad_pattern
        self.bm25_flatness_threshold = bm25_flatness_threshold

    def classify(
        self,
        query: str,
        bm25_retriever: Any | None = None,
        generator: Any | None = None,
    ) -> QueryIntent:
        """
        Classify a query into QueryIntent.NARROW or QueryIntent.BROAD.

        Args:
            query: User's question.
            bm25_retriever: Optional BM25Retriever instance for score-shape analysis.
            generator: Optional AnswerGenerator for LLM fallback classification.

        Returns:
            QueryIntent.NARROW or QueryIntent.BROAD
        """
        query_text = query.strip()
        if not query_text:
            return QueryIntent.NARROW

        # Step 1: Pattern match (free regex keyword pass)
        if self.broad_pattern.search(query_text):
            logger.info(
                f"QueryRouter: Pattern match triggered BROAD intent for '{query_text}'"
            )
            return QueryIntent.BROAD

        # Step 2: BM25 score-shape heuristic fallback (free)
        if bm25_retriever is not None and getattr(bm25_retriever, "corpus_size", 0) > 3:
            try:
                bm25_scores = bm25_retriever.get_scores(query_text)
                positive_scores = sorted(
                    [s for s in bm25_scores if s > 0], reverse=True
                )
                if positive_scores and len(positive_scores) >= 3:
                    top_score = positive_scores[0]
                    top_5_mean = sum(positive_scores[:5]) / min(5, len(positive_scores))
                    ratio = top_score / (top_5_mean + 1e-6)

                    # Flat distribution with low top score indicates broad query
                    if top_score < 3.0 and ratio < self.bm25_flatness_threshold:
                        logger.info(
                            f"QueryRouter: BM25 score-shape analysis triggered BROAD intent "
                            f"(top_score={top_score:.2f}, ratio={ratio:.2f})"
                        )
                        return QueryIntent.BROAD
            except Exception as e:
                logger.warning(f"QueryRouter BM25 score-shape check failed: {e}")

        # Step 3: Ambiguous fallback (default to NARROW to avoid unnecessary LLM calls)
        logger.debug(f"QueryRouter: Defaulting to NARROW intent for '{query_text}'")
        return QueryIntent.NARROW
