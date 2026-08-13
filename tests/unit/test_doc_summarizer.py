"""
Unit tests for DocumentSummarizer module.
"""

from unittest.mock import MagicMock

from generation.doc_summarizer import DocumentSummarizer, compute_content_hash
from utils.models import Chunk


def test_compute_content_hash():
    c1 = Chunk(text="Hello world", source="test.txt", chunk_id="c1", page=1)
    c2 = Chunk(text="Second chunk", source="test.txt", chunk_id="c2", page=1)

    hash1 = compute_content_hash([c1, c2])
    hash2 = compute_content_hash([c1, c2])
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 length


def test_doc_summarizer_cache_hit(tmp_path):
    summarizer = DocumentSummarizer(cache_dir=tmp_path)
    c1 = Chunk(text="Sample doc content", source="test.txt", chunk_id="c1", page=1)

    mock_generator = MagicMock()
    mock_generator.generate_summary_raw.return_value = "Generated Summary Output"

    # First call: cache miss -> calls LLM generator
    sum1 = summarizer.get_or_build_doc_summary([c1], mock_generator)
    assert sum1 == "Generated Summary Output"
    assert mock_generator.generate_summary_raw.call_count == 1

    # Second call with same chunks: cache hit -> 0 LLM calls!
    sum2 = summarizer.get_or_build_doc_summary([c1], mock_generator)
    assert sum2 == "Generated Summary Output"
    assert mock_generator.generate_summary_raw.call_count == 1  # Still 1 call!


def test_doc_summarizer_multi_chunk_map_reduce(tmp_path):
    summarizer = DocumentSummarizer(cache_dir=tmp_path)
    chunks = [
        Chunk(
            text=f"Chunk content number {i}",
            source="large.txt",
            chunk_id=f"c{i}",
            page=i,
        )
        for i in range(1, 10)  # 9 chunks triggers Map-Reduce loop (>5 chunks)
    ]

    mock_generator = MagicMock()
    mock_generator.generate_summary_raw.side_effect = (
        lambda prompt: f"Summary of: {prompt[:30]}"
    )

    summary = summarizer.get_or_build_doc_summary(
        chunks, mock_generator, tenant_id="test_tenant"
    )
    assert len(summary) > 0
    # MAP step (3 batches of ~4 chunks) + REDUCE step (1 call) = 4 calls total
    assert mock_generator.generate_summary_raw.call_count == 4


def test_doc_summarizer_empty_chunks(tmp_path):
    summarizer = DocumentSummarizer(cache_dir=tmp_path)
    mock_generator = MagicMock()
    result = summarizer.get_or_build_doc_summary([], mock_generator)
    assert result == "No content available to summarize."
