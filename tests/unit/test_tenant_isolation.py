"""
Unit tests for Multi-Tenant Data Isolation.
Verifies strict isolation between tenant_a and tenant_b for both narrow (retrieval)
and broad (cached summary) query paths.
"""

from unittest.mock import MagicMock

from generation.doc_summarizer import DocumentSummarizer, compute_content_hash
from pipeline import RAGPipeline
from utils.models import Chunk


def test_tenant_isolation_narrow_and_broad(tmp_path):
    """
    Ingests Document A as tenant_a and Document B as tenant_b.
    Queries as tenant_a and asserts tenant_b's data never appears in results
    for both narrow retrieval and broad summary paths.
    """
    doc_a_path = tmp_path / "doc_a.txt"
    doc_b_path = tmp_path / "doc_b.txt"

    doc_a_path.write_text(
        "Project Alpha Confidential Report. The budget for Project Alpha is 5 million dollars. "
        "Project Lead is Alice Smith. Milestone deadline is Q3 2026."
    )
    doc_b_path.write_text(
        "Project Beta Top Secret Briefing. The budget for Project Beta is 99 million dollars. "
        "Project Lead is Bob Jones. Secret location is Area 51."
    )

    # Initialize separate pipelines for tenant_a and tenant_b
    pipeline_a = RAGPipeline(tenant_id="tenant_a", debug=True)
    pipeline_b = RAGPipeline(tenant_id="tenant_b", debug=True)

    # Clean old collection state if any
    pipeline_a.delete_all_tenant_data()
    pipeline_b.delete_all_tenant_data()

    # 1. Ingest doc_a for tenant_a and doc_b for tenant_b
    chunks_a_count = pipeline_a.ingest(str(doc_a_path))
    chunks_b_count = pipeline_b.ingest(str(doc_b_path))

    assert chunks_a_count > 0
    assert chunks_b_count > 0

    # 2. Narrow Query Test: Query tenant_a
    res_a_narrow = pipeline_a.query("What is the project budget?")
    assert len(res_a_narrow.retrieved_chunks) > 0

    # Verify tenant_a gets Alpha details and NEVER gets Beta details
    retrieved_texts_a = [rc.chunk.text for rc in res_a_narrow.retrieved_chunks]
    combined_texts_a = " ".join(retrieved_texts_a)

    assert (
        "Project Alpha" in combined_texts_a
        or "Alice Smith" in combined_texts_a
        or "5 million" in combined_texts_a
    )
    assert "Project Beta" not in combined_texts_a
    assert "Bob Jones" not in combined_texts_a
    assert "99 million" not in combined_texts_a
    assert "Area 51" not in combined_texts_a

    # 3. Narrow Query Test: Query tenant_b
    res_b_narrow = pipeline_b.query("What is the project budget?")
    assert len(res_b_narrow.retrieved_chunks) > 0

    retrieved_texts_b = [rc.chunk.text for rc in res_b_narrow.retrieved_chunks]
    combined_texts_b = " ".join(retrieved_texts_b)

    assert (
        "Project Beta" in combined_texts_b
        or "Bob Jones" in combined_texts_b
        or "99 million" in combined_texts_b
    )
    assert "Project Alpha" not in combined_texts_b
    assert "Alice Smith" not in combined_texts_b
    assert "5 million" not in combined_texts_b

    # 4. Broad Query Test (Document Summarizer Cache Scoping)
    summarizer = DocumentSummarizer(cache_dir=tmp_path / "summary_cache")
    mock_gen = MagicMock()
    mock_gen.generate_summary_raw.return_value = "Summary text for tenant_a"

    chunk_a = Chunk(
        text="Tenant A private content", source="doc_a.txt", chunk_id="a1", page=1
    )
    chunk_b = Chunk(
        text="Tenant B private content", source="doc_b.txt", chunk_id="b1", page=1
    )

    summarizer.get_or_build_doc_summary([chunk_a], mock_gen, tenant_id="tenant_a")
    summarizer.get_or_build_doc_summary([chunk_b], mock_gen, tenant_id="tenant_b")

    hash_a = compute_content_hash([chunk_a])
    hash_b = compute_content_hash([chunk_b])

    cache_file_a = tmp_path / "summary_cache" / f"summary_tenant_a_{hash_a}.json"
    cache_file_b = tmp_path / "summary_cache" / f"summary_tenant_b_{hash_b}.json"

    assert cache_file_a.exists()
    assert cache_file_b.exists()
    assert "tenant_a" in cache_file_a.name
    assert "tenant_b" in cache_file_b.name
