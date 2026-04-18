"""
RAG Pipeline Orchestrator
The central facade that wires together every subsystem:
  Ingestion → Chunking → Embeddings → Vector Store → Hybrid Retrieval → Generation

Typical usage:
    pipeline = RAGPipeline()
    pipeline.ingest("docs/manual.pdf")
    response = pipeline.query("What is the return policy?")
    print(response.answer)
"""

import time
from pathlib import Path
from typing import List, Optional

from chunking import SemanticChunker
from config import retrieval_config
from embeddings import EmbeddingEngine
from generation import AnswerGenerator
from generation.diagram_generator import DiagramGenerator, DiagramResponse, detect_diagram_type
from ingestion import DocumentIngestionPipeline
from retrieval import (
    BM25Retriever,
    ChromaVectorStore,
    CrossEncoderReranker,
    HybridRetriever,
)
from utils.helpers import format_citations
from utils.logger import logger
from utils.models import Chunk, RAGResponse


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    All components are initialized lazily on first use to keep construction fast.

    Args:
        debug: If True, increase logging verbosity.
    """

    def __init__(self, debug: bool = False):
        if debug:
            import logging
            logging.getLogger("rag").setLevel(logging.DEBUG)

        logger.info("Initializing RAG Pipeline…")

        # Component initialization
        self._ingestion = DocumentIngestionPipeline()
        self._chunker = SemanticChunker()
        self._embedder = EmbeddingEngine()
        self._vector_store = ChromaVectorStore()

        # BM25 index (kept in memory alongside vector store)
        self._bm25 = BM25Retriever()

        # Reranker (optional — skip if model unavailable)
        self._reranker: Optional[CrossEncoderReranker] = None
        if retrieval_config.use_reranker:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"Reranker unavailable, skipping: {e}")

        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            embed_fn=self._embedder.embed_query,
            bm25_retriever=self._bm25,
            reranker=self._reranker,
        )

        self._generator = AnswerGenerator()
        self._diagram_generator = DiagramGenerator()

        # Track all chunks for BM25 index rebuilding
        self._all_chunks: List[Chunk] = []

        logger.info("RAG Pipeline ready")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, source: str) -> int:
        """
        Ingest a single document file into the knowledge base.

        Args:
            source: Path to the document (PDF / TXT / Markdown).

        Returns:
            Number of new chunks added.
        """
        start = time.perf_counter()
        logger.info(f"=== Ingesting: {source} ===")

        # Load
        documents = self._ingestion.ingest(source)
        if not documents:
            logger.warning(f"No content extracted from {source}")
            return 0

        # Chunk
        chunks = self._chunker.chunk(documents)
        if not chunks:
            logger.warning(f"No chunks produced from {source}")
            return 0

        # Embed
        embeddings = self._embedder.embed_chunks(chunks)

        # Store in vector DB
        self._vector_store.add_chunks(chunks, embeddings)

        # Update BM25 index
        self._all_chunks.extend(chunks)
        self._bm25.build(self._all_chunks)

        elapsed = time.perf_counter() - start
        logger.info(
            f"Ingestion complete: {len(chunks)} chunks in {elapsed:.2f}s"
        )
        return len(chunks)

    def ingest_directory(self, directory: str, recursive: bool = True) -> int:
        """
        Ingest all supported documents in a directory.

        Args:
            directory: Path to directory.
            recursive: Search subdirectories.

        Returns:
            Total number of new chunks added.
        """
        logger.info(f"=== Ingesting directory: {directory} ===")
        documents = self._ingestion.ingest_directory(directory, recursive)
        if not documents:
            return 0

        chunks = self._chunker.chunk(documents)
        if not chunks:
            return 0

        embeddings = self._embedder.embed_chunks(chunks)
        self._vector_store.add_chunks(chunks, embeddings)

        self._all_chunks.extend(chunks)
        self._bm25.build(self._all_chunks)

        logger.info(f"Directory ingestion complete: {len(chunks)} total chunks")
        return len(chunks)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Answer a question using the full RAG pipeline.

        Args:
            question: Natural language question from the user.

        Returns:
            RAGResponse with answer, citations, and source chunks.
        """
        start = time.perf_counter()
        logger.info(f"=== Query: '{question}' ===")

        # Retrieve relevant chunks
        retrieved = self._retriever.retrieve(question)

        # Generate answer
        response = self._generator.generate(question, retrieved)

        elapsed = time.perf_counter() - start
        logger.info(f"Query answered in {elapsed:.2f}s | fallback={response.is_fallback}")

        return response

    def generate_diagram(self, question: str) -> DiagramResponse:
        """
        Generate a Mermaid diagram from the knowledge base based on user request.

        Args:
            question: Natural language request, e.g. "draw a flowchart of login process"

        Returns:
            DiagramResponse with mermaid_code ready to render.
        """
        start = time.perf_counter()
        logger.info(f"=== Diagram Request: '{question}' ===")

        # Retrieve relevant chunks
        retrieved = self._retriever.retrieve(question)

        # Generate diagram
        result = self._diagram_generator.generate(question, retrieved)

        elapsed = time.perf_counter() - start
        logger.info(f"Diagram generated in {elapsed:.2f}s | fallback={result.is_fallback}")

        return result

    def is_diagram_request(self, question: str) -> bool:
        """Return True if the question is asking for a visual diagram."""
        return detect_diagram_type(question) is not None

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return basic stats about the current knowledge base."""
        return {
            "total_chunks_in_vector_store": self._vector_store.count(),
            "total_chunks_in_bm25": self._bm25.corpus_size,
            "embedding_model": self._embedder.model_name,
        }

    def delete_document(self, source: str) -> int:
        """Remove all chunks for a specific source document."""
        deleted = self._vector_store.delete_by_source(source)
        # Rebuild BM25 without the deleted source
        self._all_chunks = [c for c in self._all_chunks if c.source != source]
        if self._all_chunks:
            self._bm25.build(self._all_chunks)
        logger.info(f"Deleted {deleted} chunks for '{source}'")
        return deleted
