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

from chunking import SemanticChunker
from config import retrieval_config
from embeddings import EmbeddingEngine
from generation import AnswerGenerator, doc_summarizer
from generation.diagram_generator import (
    DiagramGenerator,
    DiagramResponse,
    detect_diagram_type,
)
from ingestion import DocumentIngestionPipeline
from retrieval import (
    BM25Retriever,
    ChromaVectorStore,
    CrossEncoderReranker,
    HybridRetriever,
    QueryIntent,
    QueryRouter,
)
from utils.logger import logger
from utils.models import Chunk, RAGResponse
from utils.rate_limiter import rate_limiter


def is_summary_query(question: str) -> bool:
    """
    Detect broad summary or document overview requests using heuristic matching.
    Matches queries like 'explain the document', 'summarize', 'give an overview', 'what is this about'.
    """
    q_lower = question.lower().strip()
    summary_triggers = [
        "explain the document",
        "explain document",
        "explain pdf",
        "summarize the document",
        "summarize pdf",
        "summarize",
        "summarise",
        "give an overview",
        "give overview",
        "what is this about",
        "what is the document about",
        "what is this pdf about",
        "document summary",
        "pdf summary",
        "explain the pitch deck",
        "explain pitch deck",
    ]
    return any(trigger in q_lower for trigger in summary_triggers) or q_lower in [
        "summary",
        "overview",
        "explain",
    ]


class RAGPipeline:
    """
    High-level facade orchestrating document ingestion, retrieval, reranking,
    answer generation, and diagram creation.

    Per-Tenant Data Isolation:
    Each instance is bound to a specific `tenant_id`, using a separate ChromaDB collection
    (`tenant_{tenant_id}`) and an isolated per-tenant BM25 index.

    Args:
        tenant_id: Unique identifier for the tenant (default: "default_tenant").
        user_id:   Backwards-compatibility alias for tenant_id.
        debug:     If True, increase logging verbosity.
    """

    def __init__(
        self,
        tenant_id: str | None = None,
        user_id: str | None = None,
        debug: bool = False,
    ):
        eff_tenant = tenant_id or user_id or "default_tenant"
        if debug:
            import logging

            logging.getLogger("rag").setLevel(logging.DEBUG)

        self.tenant_id = eff_tenant
        self.user_id = eff_tenant
        logger.info(f"Initializing RAG Pipeline for tenant_id='{eff_tenant}'…")

        # Component initialization with per-tenant data isolation
        self._ingestion = DocumentIngestionPipeline()
        self._chunker = SemanticChunker()
        self._embedder = EmbeddingEngine()
        self._vector_store = ChromaVectorStore(tenant_id=eff_tenant)

        # Per-tenant BM25 index (built strictly from tenant's chunks)
        self._bm25 = BM25Retriever()
        self._all_chunks: list[Chunk] = self._vector_store.get_all_chunks()
        if self._all_chunks:
            self._bm25.build(self._all_chunks)

        # Reranker (optional — skip if model unavailable)
        self._reranker: CrossEncoderReranker | None = None
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
        self._query_router = QueryRouter()

        logger.info(
            f"RAG Pipeline ready for tenant_id='{eff_tenant}' — {len(self._all_chunks)} chunk(s) indexed"
        )

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
        logger.info(f"=== Ingesting for tenant '{self.tenant_id}': {source} ===")

        # Load
        documents = self._ingestion.ingest(source, tenant_id=self.tenant_id)
        if not documents:
            logger.warning(f"No content extracted from {source}")
            return 0

        # Chunk
        chunks = self._chunker.chunk(documents, tenant_id=self.tenant_id)
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
        logger.info(f"Ingestion complete: {len(chunks)} chunks in {elapsed:.2f}s")
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
        logger.info(
            f"=== Ingesting directory for tenant '{self.tenant_id}': {directory} ==="
        )
        documents = self._ingestion.ingest_directory(
            directory, recursive=recursive, tenant_id=self.tenant_id
        )
        if not documents:
            return 0

        chunks = self._chunker.chunk(documents, tenant_id=self.tenant_id)
        if not chunks:
            return 0

        embeddings = self._embedder.embed_chunks(chunks)
        self._vector_store.add_chunks(chunks, embeddings)

        self._all_chunks.extend(chunks)
        self._bm25.build(self._all_chunks)

        logger.info(f"Directory ingestion complete: {len(chunks)} total chunks")
        return len(chunks)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        anonymize_pii: bool = False,
    ) -> RAGResponse:
        """
        Answer a question using the full RAG pipeline with optional ML & Privacy features.

        Args:
            question: Natural language question from the user.
            use_hyde: Enable HyDE (Hypothetical Document Embeddings) retrieval.
            use_multi_query: Enable Multi-Query Expansion retrieval.
            provider: Optional LLM provider override ('groq', 'ollama', 'openai', etc.).
            model: Optional model name override.
            api_key: Optional custom API key (Bring Your Own Key - BYOK).
            anonymize_pii: If True, redacts PII before sending context to LLM.

        Returns:
            RAGResponse with answer, citations, ML scores, and metadata.
        """
        start = time.perf_counter()
        rate_limiter.check_rate_limit(self.user_id)
        logger.info(
            f"=== Query: '{question}' (HyDE={use_hyde}, MultiQuery={use_multi_query}, provider={provider}, anonymize_pii={anonymize_pii}) ==="
        )

        generator = self._generator
        if provider or model or api_key:
            try:
                generator = AnswerGenerator(
                    provider=provider or self._generator.provider,
                    model=model or self._generator.model,
                    api_key=api_key,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to instantiate custom generator ({e}), falling back to default: {e}"
                )
                generator = self._generator

        # Query Intent Classification (NARROW vs BROAD)
        intent = self._query_router.classify(
            question, bm25_retriever=self._bm25, generator=generator
        )

        if intent == QueryIntent.BROAD or is_summary_query(question):
            logger.info(
                f"Broad summary query detected (intent={intent}) — routing to Cached Document Summarizer path"
            )
            # Retrieve or build cached document summary
            doc_summary_text = doc_summarizer.get_or_build_doc_summary(
                self._all_chunks, generator=generator
            )
            summary_chunk = Chunk(
                text=doc_summary_text,
                source="cached_doc_summary",
                chunk_id="doc_summary_cached",
                page=1,
            )
            ordered_chunks = self._retriever.get_ordered_document_chunks() or [
                summary_chunk
            ]
            response = generator.generate_summary(
                question, ordered_chunks, anonymize_pii=anonymize_pii
            )
            elapsed = time.perf_counter() - start
            logger.info(
                f"Broad summary query answered in {elapsed:.2f}s | fallback={response.is_fallback}"
            )
            return response

        hyde_doc = ""

        expanded_queries = []
        search_query = question

        # ML Feature 1: HyDE Retrieval
        if use_hyde:
            logger.info("Generating hypothetical document for HyDE...")
            hyde_doc = generator.generate_hyde_doc(question)
            if hyde_doc:
                search_query = f"{question}\n{hyde_doc}"

        # ML Feature 2: Multi-Query Expansion
        if use_multi_query:
            logger.info("Generating query expansions...")
            expanded_queries = generator.generate_query_expansions(question)
            all_chunks = self._retriever.retrieve(search_query)
            for eq in expanded_queries:
                extra = self._retriever.retrieve(eq)
                existing_ids = {rc.chunk.chunk_id for rc in all_chunks}
                for rc in extra:
                    if rc.chunk.chunk_id not in existing_ids:
                        all_chunks.append(rc)
                        existing_ids.add(rc.chunk.chunk_id)
            retrieved_pool = all_chunks
        else:
            retrieved_pool = self._retriever.retrieve(search_query)

        # For broad queries (e.g., "explain pitch deck", "summary"), expand context limit
        is_broad = any(
            k in question.lower()
            for k in (
                "pitch deck",
                "explain",
                "summarize",
                "overview",
                "summary",
                "all",
                "deck",
            )
        )
        max_chunks = 8 if is_broad else self._retriever.top_n_rerank
        retrieved = retrieved_pool[:max_chunks]

        # Generate answer & compute Self-RAG metrics
        response = generator.generate(question, retrieved, anonymize_pii=anonymize_pii)
        response.hyde_document = hyde_doc
        response.expanded_queries = expanded_queries

        elapsed = time.perf_counter() - start
        logger.info(
            f"Query answered in {elapsed:.2f}s | fallback={response.is_fallback} | "
            f"faithfulness={response.faithfulness_score:.2f} | relevance={response.relevance_score:.2f}"
        )

        return response
        response.hyde_document = hyde_doc
        response.expanded_queries = expanded_queries

        elapsed = time.perf_counter() - start
        logger.info(
            f"Query answered in {elapsed:.2f}s | fallback={response.is_fallback} | "
            f"faithfulness={response.faithfulness_score:.2f} | relevance={response.relevance_score:.2f}"
        )

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
        logger.info(
            f"Diagram generated in {elapsed:.2f}s | fallback={result.is_fallback}"
        )

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

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk by chunk_id."""
        deleted = self._vector_store.delete_by_id(chunk_id)
        if deleted:
            self._all_chunks = [c for c in self._all_chunks if c.chunk_id != chunk_id]
            if self._all_chunks:
                self._bm25.build(self._all_chunks)
        return deleted

    def get_all_chunks(self) -> list[Chunk]:
        """Return all chunks stored in the vector database."""
        return self._vector_store.get_all_chunks()

    def delete_all_tenant_data(self) -> None:
        """Drop the entire ChromaDB collection for this tenant and clear local caches."""
        self._vector_store.delete_tenant_collection()
        self._all_chunks = []
        self._bm25.build([])
        logger.warning(
            f"All data for tenant '{self.tenant_id}' has been permanently deleted."
        )

    def reset_database(self) -> None:
        """Reset the vector database and clear all stored knowledge."""
        self._vector_store.reset()
        self._all_chunks = []
        self._bm25.build([])
        logger.warning("Vector store and BM25 index completely reset.")
