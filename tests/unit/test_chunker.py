import pytest

from chunking.chunker import SemanticChunker
from utils.helpers import token_count_approx
from utils.models import Document


def test_empty_or_whitespace_input_produces_zero_chunks():
    chunker = SemanticChunker(min_chunk_size=1)
    empty_doc = Document(content="", source="test.txt")
    space_doc = Document(content="   \n\t   ", source="test.txt")

    assert chunker.chunk([empty_doc]) == []
    assert chunker.chunk([space_doc]) == []


def test_chunk_sizes_within_bounds():
    chunker = SemanticChunker(chunk_size=20, chunk_overlap=5, min_chunk_size=5)
    content = (
        "First sentence in the test document. "
        "Second sentence is also quite informative and clear. "
        "Third sentence continues the story with more details. "
        "Fourth sentence concludes the section cleanly."
    )
    doc = Document(content=content, source="doc.txt")
    chunks = chunker.chunk([doc])

    assert len(chunks) > 0
    for chunk in chunks:
        tokens = token_count_approx(chunk.text)
        assert tokens >= chunker.min_chunk_size
        assert tokens <= chunker.chunk_size * 1.5


def test_overlap_between_consecutive_chunks():
    chunker = SemanticChunker(chunk_size=15, chunk_overlap=8, min_chunk_size=1)
    sentences = [
        "Alpha sentence goes first here.",
        "Beta sentence follows right after.",
        "Gamma sentence comes in third position.",
        "Delta sentence ends the list.",
    ]
    doc = Document(content=" ".join(sentences), source="overlap.txt")
    chunks = chunker.chunk([doc])

    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        c1_text = chunks[i].text
        c2_text = chunks[i + 1].text
        c1_words = set(c1_text.split())
        c2_words = set(c2_text.split())
        overlap = c1_words.intersection(c2_words)
        assert len(overlap) > 0


def test_chunking_never_splits_mid_sentence():
    chunker = SemanticChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=1)
    sentences = [
        "First sentence is complete.",
        "Second sentence is also complete.",
        "Third sentence finishes here.",
        "Fourth sentence stands alone.",
    ]
    doc = Document(content=" ".join(sentences), source="sentences.txt")
    chunks = chunker.chunk([doc])

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.text.endswith(".")


def test_oversized_sentence_splitting():
    chunker = SemanticChunker(chunk_size=10, chunk_overlap=2, min_chunk_size=1)
    long_sentence = " ".join([f"word{i}" for i in range(50)]) + "."
    doc = Document(
        content=f"Short intro sentence. {long_sentence} Final ending sentence.",
        source="long.txt",
    )
    chunks = chunker.chunk([doc])

    assert len(chunks) > 1


def test_chunker_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=10, chunk_overlap=15)
