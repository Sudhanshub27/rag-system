"""
Integration Test: Per-User Multi-Tenant Data Isolation

Verifies structural data isolation across separate ChromaDB collections and BM25 indices:
1. User A (Alice) ingests secret Document A.
2. User B (Bob) ingests secret Document B.
3. Confirms User A's queries return ONLY Document A and zero chunks from Document B.
4. Confirms User B's queries return ONLY Document B and zero chunks from Document A.
5. Confirms deleting User A's document has zero impact on User B's knowledge base.
"""

from pathlib import Path

import pytest

from pipeline import RAGPipeline


@pytest.fixture
def temp_user_docs(tmp_path):
    """Create distinct test documents for two separate tenants."""
    doc_a = tmp_path / "alice_project_apollo.txt"
    doc_a.write_text(
        "Project Apollo Confidential Launch Specs. Launch date is March 15th. "
        "The primary propulsion engine uses liquid methane fuel cells."
    )

    doc_b = tmp_path / "bob_project_zeus.txt"
    doc_b.write_text(
        "Project Zeus Financial Budget Audit. Total budget allocated is 15 million USD. "
        "Primary expense vendor is Acme Heavy Industries."
    )

    return str(doc_a), str(doc_b)


def test_per_user_data_isolation(temp_user_docs, tmp_path):
    doc_a_path, doc_b_path = temp_user_docs

    user_a_id = "test_user_alice_123"
    user_b_id = "test_user_bob_456"

    # Initialize isolated pipelines
    pipeline_a = RAGPipeline(user_id=user_a_id)
    pipeline_b = RAGPipeline(user_id=user_b_id)

    # Clear any previous test data
    pipeline_a.reset_database()
    pipeline_b.reset_database()

    # Step 1: Ingest distinct documents for each user
    n_a = pipeline_a.ingest(doc_a_path)
    n_b = pipeline_b.ingest(doc_b_path)

    assert n_a > 0, "Alice should ingest at least 1 chunk"
    assert n_b > 0, "Bob should ingest at least 1 chunk"

    # Step 2: Check vector store collection isolation
    chunks_a = pipeline_a.get_all_chunks()
    chunks_b = pipeline_b.get_all_chunks()

    assert len(chunks_a) == n_a
    assert len(chunks_b) == n_b

    # Assert no cross-contamination in stored chunk sources
    sources_a = {c.source for c in chunks_a}
    sources_b = {c.source for c in chunks_b}

    assert Path(doc_a_path).name in sources_a
    assert Path(doc_b_path).name not in sources_a

    assert Path(doc_b_path).name in sources_b
    assert Path(doc_a_path).name not in sources_b

    # Step 3: Query Alice's pipeline for Bob's secret ("Zeus budget")
    res_a = pipeline_a.query("What is the budget for Project Zeus?")
    for rc in res_a.retrieved_chunks:
        assert (
            "Zeus" not in rc.chunk.text
        ), "Alice must NEVER retrieve Bob's Zeus chunks!"
        assert (
            "Acme Heavy" not in rc.chunk.text
        ), "Alice must NEVER retrieve Bob's vendors!"

    # Step 4: Query Bob's pipeline for Alice's secret ("Apollo launch engine")
    res_b = pipeline_b.query("What propulsion fuel does Project Apollo use?")
    for rc in res_b.retrieved_chunks:
        assert (
            "Apollo" not in rc.chunk.text
        ), "Bob must NEVER retrieve Alice's Apollo chunks!"
        assert (
            "methane" not in rc.chunk.text
        ), "Bob must NEVER retrieve Alice's propulsion specs!"

    # Step 5: Verify Alice's query for Apollo returns her own chunk
    res_a_own = pipeline_a.query("What propulsion engine does Apollo use?")
    assert len(res_a_own.retrieved_chunks) > 0
    assert any("methane" in rc.chunk.text.lower() for rc in res_a_own.retrieved_chunks)

    # Step 6: Verify deletion isolation — deleting Alice's doc does not alter Bob's data
    deleted_a = pipeline_a.delete_document(Path(doc_a_path).name)
    assert deleted_a == n_a
    assert len(pipeline_a.get_all_chunks()) == 0

    # Bob's data must remain completely intact
    assert len(pipeline_b.get_all_chunks()) == n_b

    # Clean up test user collections
    pipeline_a.reset_database()
    pipeline_b.reset_database()
