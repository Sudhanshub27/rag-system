"""
Hybrid Retrieval Engine
Combines BM25 keyword search and vector similarity search using
Reciprocal Rank Fusion (RRF), then re-ranks with a cross-encoder.

Pipeline:
  1. BM25  → top-K keyword candidates
  2. Vector → top-K semantic candidates
  3. RRF   → merge + deduplicate by chunk_id
  4. Cross-Encoder → rerank merged candidates
  5. Return top-N
"""

from typing import Dict, List, Optional

from config import retrieval_config
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import ChromaVectorStore
from utils.logger import logger
from utils.models import Chunk, RetrievedChunk


class HybridRetriever:
    """
    Orchestrates hybrid (BM25 + vector) retrieval with optional reranking.

    Args:
        vector_store:   Initialized ChromaVectorStore.
        embed_fn:       Callable that takes a str and returns List[float].
        bm25_retriever: Optional BM25Retriever (created internally if None).
        reranker:       Optional CrossEncoderReranker.
        top_k:          Number of candidates to fetch from each source.
        top_n_rerank:   Final number of chunks to return.
        bm25_weight:    Score weight for BM25 results in RRF fusion.
        vector_weight:  Score weight for vector results in RRF fusion.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_fn,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k: int = retrieval_config.top_k,
        top_n_rerank: int = retrieval_config.top_n_rerank,
        bm25_weight: float = retrieval_config.bm25_weight,
        vector_weight: float = retrieval_config.vector_weight,
    ):
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker
        self.top_k = top_k
        self.top_n_rerank = top_n_rerank
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    # ── Public API ────────────────────────────────────────────────────────────

    def update_bm25(self, chunks: List[Chunk]) -> None:
        """Rebuild the BM25 index when the corpus changes."""
        self.bm25_retriever.build(chunks)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: User question string.

        Returns:
            List of RetrievedChunk objects, sorted by final relevance score.
        """
        logger.info(f"Retrieval query: '{query[:80]}...' " if len(query) > 80 else f"Retrieval query: '{query}'")

        # Stage 1: BM25 keyword retrieval
        bm25_results: List[RetrievedChunk] = []
        if retrieval_config.use_bm25 and self.bm25_retriever.corpus_size > 0:
            bm25_results = self.bm25_retriever.query(query, top_k=self.top_k)
            logger.debug(f"BM25 returned {len(bm25_results)} chunks")

        # Stage 2: Vector similarity retrieval
        query_embedding = self.embed_fn(query)
        vector_results = self.vector_store.query(query_embedding, top_k=self.top_k)
        logger.debug(f"Vector search returned {len(vector_results)} chunks")

        # Stage 3: Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(bm25_results, vector_results)
        logger.debug(f"RRF merged {len(merged)} unique chunks")

        if not merged:
            logger.warning("Retrieval returned no results")
            return []

        # Stage 4: Cross-encoder reranking
        if self.reranker and retrieval_config.use_reranker:
            final = self.reranker.rerank(query, merged)
        else:
            final = sorted(merged, key=lambda x: x.score, reverse=True)
            final = final[: self.top_n_rerank]
            for rank, rc in enumerate(final, start=1):
                rc.rank = rank

        logger.info(
            f"Final retrieval: {len(final)} chunk(s) returned. "
            f"Top score: {final[0].score:.4f}" if final else "No chunks"
        )
        return final

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[RetrievedChunk],
        vector_results: List[RetrievedChunk],
        k: int = 60,  # RRF constant (k=60 is the standard default)
    ) -> List[RetrievedChunk]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ (weight / (k + rank))

        This is list-rank-based, so original scores need not be on the
        same scale — it handles the BM25 vs cosine scale mismatch cleanly.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        def _update(results: List[RetrievedChunk], weight: float) -> None:
            for rank, rc in enumerate(results, start=1):
                cid = rc.chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight / (k + rank)
                if cid not in chunk_map:
                    chunk_map[cid] = rc

        _update(bm25_results, self.bm25_weight)
        _update(vector_results, self.vector_weight)

        # Sort by fused score and return
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)  # type: ignore
        merged = []
        for cid in sorted_ids:
            rc = chunk_map[cid]
            rc.score = rrf_scores[cid]
            merged.append(rc)

        return merged
