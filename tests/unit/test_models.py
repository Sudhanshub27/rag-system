from utils.models import Chunk, Document, RAGResponse, RetrievedChunk


def test_document_defaults():
    doc = Document(content="Sample content", source="sample.pdf")
    assert doc.content == "Sample content"
    assert doc.source == "sample.pdf"
    assert doc.metadata == {}


def test_chunk_defaults():
    chunk = Chunk(text="Sample text", source="sample.pdf", chunk_id="cid1")
    assert chunk.text == "Sample text"
    assert chunk.source == "sample.pdf"
    assert chunk.chunk_id == "cid1"
    assert chunk.page == 0
    assert chunk.metadata == {}


def test_retrieved_chunk_defaults():
    chunk = Chunk(text="Sample text", source="sample.pdf", chunk_id="cid1")
    rc = RetrievedChunk(chunk=chunk, score=0.95)
    assert rc.chunk == chunk
    assert rc.score == 0.95
    assert rc.rank == 0


def test_rag_response_defaults_and_fallback():
    resp = RAGResponse(
        answer="Sample answer",
        citations=["[1] Source: sample.pdf, Page: 1"],
        retrieved_chunks=[],
        query="Sample query",
    )
    assert resp.answer == "Sample answer"
    assert resp.citations == ["[1] Source: sample.pdf, Page: 1"]
    assert resp.retrieved_chunks == []
    assert resp.query == "Sample query"
    assert resp.is_fallback is False

    fallback_resp = RAGResponse(
        answer="I don't know.",
        citations=[],
        retrieved_chunks=[],
        query="Unknown query",
        is_fallback=True,
    )
    assert fallback_resp.is_fallback is True
