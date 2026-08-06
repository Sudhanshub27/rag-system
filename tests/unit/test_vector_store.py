import pytest

from retrieval.vector_store import ChromaVectorStore
from utils.models import Chunk


def test_vector_store_add_query_delete_count_reset(temp_chroma_dir):
    store = ChromaVectorStore(
        persist_directory=temp_chroma_dir, collection_name="test_col"
    )

    assert store.count() == 0

    chunk1 = Chunk(
        text="Vector search is fast",
        source="doc1.pdf",
        chunk_id="id1",
        page=1,
        metadata={"category": "ai"},
    )
    chunk2 = Chunk(
        text="BM25 search is sparse",
        source="doc2.pdf",
        chunk_id="id2",
        page=2,
        metadata={"category": "search"},
    )
    embeddings = [[0.1] * 384, [0.2] * 384]

    store.add_chunks([chunk1, chunk2], embeddings)
    assert store.count() == 2

    # Adding same chunks should be idempotent
    store.add_chunks([chunk1], [[0.1] * 384])
    assert store.count() == 2

    # Mismatched lengths raise ValueError
    with pytest.raises(ValueError):
        store.add_chunks([chunk1], [embeddings[0], embeddings[1]])

    # Query vector store
    results = store.query(query_embedding=[0.1] * 384, top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id in ("id1", "id2")

    # Delete by source
    deleted = store.delete_by_source("doc1.pdf")
    assert deleted == 1
    assert store.count() == 1

    # Reset collection
    store.reset()
    assert store.count() == 0


def test_vector_store_empty_query(temp_chroma_dir):
    store = ChromaVectorStore(
        persist_directory=temp_chroma_dir, collection_name="test_empty"
    )
    results = store.query(query_embedding=[0.1] * 384, top_k=5)
    assert results == []
