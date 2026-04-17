"""
BM25 Retriever
Provides keyword-based sparse retrieval over in-memory chunk corpus.
Used alongside vector search for hybrid retrieval.

Library: rank_bm25 (BM25Okapi implementation)
"""

from typing import List, Optional

from utils.logger import logger
from utils.models import Chunk, RetrievedChunk


class BM25Retriever:
    """
    In-memory BM25 keyword retriever.

    Must be re-built whenever the chunk corpus changes.
    Intended to be used alongside ChromaVectorStore for hybrid retrieval.

    Args:
        chunks: List of Chunk objects to build the index over.
    """

    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self._chunks: List[Chunk] = []
        self._bm25 = None

        if chunks:
            self.build(chunks)

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build (or rebuild) the BM25 index from a list of chunks.

        Args:
            chunks: All chunks in the corpus.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank_bm25 is required for BM25 retrieval. "
                "Run: pip install rank-bm25"
            ) from e

        self._chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(chunks)} chunks")

    def query(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """
        Retrieve top-K chunks by BM25 score.

        Args:
            query: User query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by BM25 score (descending).
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 index is empty — call build() first")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Pair chunks with scores and sort
        scored = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results: List[RetrievedChunk] = []
        for idx, score in scored[:top_k]:
            if score <= 0.0:
                break  # No more relevant results
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(score),
                )
            )

        logger.debug(f"BM25 retrieved {len(results)} chunk(s) for query")
        return results

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the BM25 index."""
        return len(self._chunks)
