from retrieval.bm25_retriever import BM25Retriever
from utils.models import Chunk


def test_bm25_empty_corpus_returns_empty():
    retriever = BM25Retriever()
    assert retriever.corpus_size == 0
    assert retriever.query("test query") == []

    # Calling build with empty list
    retriever.build([])
    assert retriever.corpus_size == 0
    assert retriever.query("test query") == []


def test_bm25_exact_keyword_match_ranks_above_lexically_different():
    chunk_exact = Chunk(
        text="Quantum computing utilizes qubits for massive parallel computations.",
        source="quantum.txt",
        chunk_id="c1",
    )
    chunk_semantic = Chunk(
        text="Advanced subatomic physics allows complex mathematical calculations.",
        source="physics.txt",
        chunk_id="c2",
    )
    chunk_unrelated = Chunk(
        text="Organic gardening requires rich soil compost and regular watering schedule.",
        source="gardening.txt",
        chunk_id="c3",
    )

    retriever = BM25Retriever(chunks=[chunk_exact, chunk_semantic, chunk_unrelated])
    results = retriever.query("quantum computing qubits", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score > 0.0
