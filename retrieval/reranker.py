"""
Cross-Encoder Reranker
Scores query-chunk pairs for precise relevance after initial retrieval.

Why reranking?
  Vector search returns approximate neighbours; cross-encoders perform
  true joint query-document attention and produce far more accurate
  relevance scores, at the cost of being slower.

  Typical pipeline: retrieve K candidates → rerank → keep top N.
"""

from typing import List

from config import retrieval_config
from utils.logger import logger
from utils.models import RetrievedChunk


class CrossEncoderReranker:
    """
    Rerank retrieved chunks using a sentence-transformers cross-encoder.

    Args:
        model_name: HuggingFace cross-encoder model identifier.
        top_n:      Number of chunks to return after reranking.
    """

    def __init__(
        self,
        model_name: str = retrieval_config.reranker_model,
        top_n: int = retrieval_config.top_n_rerank,
    ):
        self.model_name = model_name
        self.top_n = top_n
        logger.info(f"Loading cross-encoder reranker: {model_name}")

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            logger.info("Cross-encoder reranker loaded successfully")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Run: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load reranker '{model_name}': {e}")
            raise

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Rerank retrieved chunks against the query.

        Args:
            query:            User query string.
            retrieved_chunks: Candidates from the first-stage retriever.
            min_score:        Filter out chunks below this reranker score.

        Returns:
            Top-N chunks sorted by cross-encoder score (descending),
            with updated rank and score attributes.
        """
        if not retrieved_chunks:
            return []

        pairs = [(query, rc.chunk.text) for rc in retrieved_chunks]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Reranker prediction failed: {e}")
            # Graceful degradation: return original order
            return retrieved_chunks[: self.top_n]

        # Attach new scores
        for rc, score in zip(retrieved_chunks, scores):
            rc.score = float(score)

        # Sort by score descending, filter, and assign ranks
        reranked = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)
        reranked = [rc for rc in reranked if rc.score >= min_score]
        reranked = reranked[: self.top_n]

        for rank, rc in enumerate(reranked, start=1):
            rc.rank = rank

        logger.debug(
            f"Reranking: {len(retrieved_chunks)} candidates → "
            f"{len(reranked)} kept (min_score={min_score})"
        )
        return reranked
