"""
Shared data models (dataclasses) used across all RAG modules.
Keeping models in one place prevents circular imports.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """
    Represents a raw document loaded from disk or URL.

    Attributes:
        content:  Full text content of the document.
        source:   File path or URL the document was loaded from.
        metadata: Arbitrary key-value pairs (page numbers, titles, etc.).
    """

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """
    A single text chunk derived from a Document.

    Attributes:
        text:       The chunk's text content.
        source:     Original document source (file path / URL).
        chunk_id:   Globally unique identifier for this chunk.
        page:       Page number in source document (if applicable).
        metadata:   Additional metadata forwarded from the parent Document.
    """

    text: str
    source: str
    chunk_id: str
    page: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """
    A chunk returned by the retrieval engine, augmented with a relevance score.

    Attributes:
        chunk:      The underlying Chunk.
        score:      Relevance score (higher = more relevant).
        rank:       Position after reranking (1-indexed).
    """

    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class RAGResponse:
    """
    Final response returned by the RAG pipeline to the caller.

    Attributes:
        answer:            Generated answer text.
        citations:         List of citation strings shown to the user.
        retrieved_chunks:  Raw chunks used to generate the answer.
        query:             The original user query.
        is_fallback:       True when context was insufficient to answer.
        faithfulness_score: Groundedness score of answer against retrieved context (0-1).
        relevance_score:   Relevance score of context against user query (0-1).
        expanded_queries:  Multi-query expansion variants used during retrieval.
        hyde_document:     Hypothetical document generated for HyDE retrieval.
    """

    answer: str
    citations: list[str]
    retrieved_chunks: list[RetrievedChunk]
    query: str
    is_fallback: bool = False
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    expanded_queries: list[str] = field(default_factory=list)
    hyde_document: str = ""
