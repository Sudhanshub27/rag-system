"""
Utility helpers shared across modules.
"""

import hashlib
import re
import unicodedata
from pathlib import Path


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
    text = "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t\r"
    )

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


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple regex heuristics.
    Used by the chunker to preserve semantic sentence boundaries.
    """
    # Split on '.', '!', '?' followed by whitespace + capital letter
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def format_citations(retrieved_chunks) -> list[str]:
    """
    Format retrieved chunks into detailed citation strings for display.

    Args:
        retrieved_chunks: List of RetrievedChunk objects.

    Returns:
        List of formatted citation strings with page numbers and text excerpts.
    """
    citations = []
    for i, rc in enumerate(retrieved_chunks, start=1):
        source = rc.chunk.source
        page = rc.chunk.page
        snippet = rc.chunk.text.replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:180] + "…"
        citations.append(f'[{i}] Source: {source}, Page: {page}\nExcerpt: "{snippet}"')
    return citations


def get_pdf_page_image(pdf_path: str, page_num: int, dpi: int = 150) -> bytes | None:
    """
    Render a specific page of a PDF file into PNG image bytes using PyMuPDF (fitz).

    Args:
        pdf_path: Path to the PDF file.
        page_num: 1-indexed page number.
        dpi: Resolution in dots per inch (default: 150).

    Returns:
        PNG image bytes or None if rendering fails.
    """
    try:
        import fitz  # PyMuPDF

        path = Path(pdf_path)
        if not path.exists():
            return None

        doc = fitz.open(str(path))
        if 1 <= page_num <= len(doc):
            page = doc[page_num - 1]
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            doc.close()
            return img_bytes
        doc.close()
    except Exception:
        pass
    return None


def sanitize_collection_name(user_id: str) -> str:
    """
    Sanitize a user identifier into a valid ChromaDB collection name.

    ChromaDB collection name rules:
    - Must be 3-63 characters long.
    - Matches ^[a-zA-Z0-9_-]+$
    - Starts and ends with an alphanumeric character.
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", str(user_id)).strip("_")
    if not clean:
        clean = "default_user"
    collection_name = f"user_{clean}"
    if len(collection_name) > 63:
        collection_name = collection_name[:63].rstrip("_")
    while len(collection_name) < 3:
        collection_name += "_usr"
    return collection_name
