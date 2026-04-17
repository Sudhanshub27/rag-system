"""
Shared data models (dataclasses) used across all RAG modules.
Keeping models in one place prevents circular imports.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    """
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunk]
    query: str
    is_fallback: bool = False
