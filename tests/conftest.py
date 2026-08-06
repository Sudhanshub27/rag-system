import pytest

from generation.answer_generator import AnswerGenerator
from utils.models import Chunk


@pytest.fixture
def sample_chunks():
    """Provides a list of sample Chunk objects for testing."""
    return [
        Chunk(
            text="Retrieval-Augmented Generation enhances language models with external knowledge.",
            source="doc_a.txt",
            chunk_id="chunk-1",
            page=1,
            metadata={"chunk_index": 0},
        ),
        Chunk(
            text="Semantic chunking preserves sentence boundaries during text splitting.",
            source="doc_a.txt",
            chunk_id="chunk-2",
            page=1,
            metadata={"chunk_index": 1},
        ),
        Chunk(
            text="BM25 is a sparse keyword retrieval algorithm based on TF-IDF scoring.",
            source="doc_b.txt",
            chunk_id="chunk-3",
            page=2,
            metadata={"chunk_index": 0},
        ),
    ]


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provides a temporary directory path for isolated Chroma DB persistence."""
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir)


@pytest.fixture
def mock_llm_call(monkeypatch, mocker):
    """
    Monkeypatches/mocks the LLM provider call and client initialization
    in AnswerGenerator so tests never make external API calls or require API keys.
    """
    mock_client = mocker.MagicMock()
    monkeypatch.setattr(AnswerGenerator, "_init_client", lambda self: mock_client)

    mock_call = mocker.MagicMock(
        return_value="This is a test generated answer based on retrieved context [1]."
    )
    monkeypatch.setattr(AnswerGenerator, "_call_llm", mock_call)

    return mock_call
