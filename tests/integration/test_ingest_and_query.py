import pytest

import config
from pipeline import RAGPipeline


@pytest.mark.integration
def test_ingest_and_query_integration(temp_chroma_dir, mock_llm_call, monkeypatch):
    monkeypatch.setattr(
        config.vector_store_config, "persist_directory", temp_chroma_dir
    )

    pipeline = RAGPipeline()
    sample_file = "docs/sample_doc.txt"
    chunks_added = pipeline.ingest(sample_file)

    assert chunks_added > 0

    response = pipeline.query("What is Retrieval-Augmented Generation?")

    assert len(response.retrieved_chunks) > 0
    top_chunk = response.retrieved_chunks[0]
    assert "sample_doc.txt" in top_chunk.chunk.source
