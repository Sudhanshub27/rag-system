"""
Utility helpers shared across modules.
"""

import hashlib
import re
import unicodedata
from typing import List


def generate_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """
    Deterministically generate a unique ID for a chunk.

    Uses a hash of (source + index + first 64 chars of text) so IDs are
    stable across runs and won't collide even for duplicate source names.
    """
    raw = f"{source}::{chunk_index}::{text[:64]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    """
    Clean and normalize raw text:
    - Normalize unicode to NFC form.
    - Remove control characters (except newlines/tabs).
    - Collapse excessive whitespace while preserving paragraph breaks.
    - Strip leading/trailing whitespace.
    """
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)

    # Remove control chars (keep \n, \t, \r)
    text = re.sub(r"[^\S\n\t\r ]+", " ", text)

    # Remove non-printable characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t\r")

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def token_count_approx(text: str) -> int:
    """
    Approximate token count using whitespace split.
    Rule of thumb: ~0.75 tokens per word for English text.
    Good enough for chunking decisions without loading a tokenizer.
    """
    words = len(text.split())
    return int(words / 0.75)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex heuristics.
    Used by the chunker to preserve semantic sentence boundaries.
    """
    # Split on '.', '!', '?' followed by whitespace + capital letter
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def format_citations(retrieved_chunks) -> List[str]:
    """
    Format retrieved chunks into citation strings for display.

    Args:
        retrieved_chunks: List of RetrievedChunk objects.

    Returns:
        List of formatted citation strings like:
        "[1] Source: file.pdf, Page: 3"
    """
    citations = []
    for i, rc in enumerate(retrieved_chunks, start=1):
        source = rc.chunk.source
        page = rc.chunk.page
        citations.append(f"[{i}] Source: {source}, Page: {page}")
    return citations
