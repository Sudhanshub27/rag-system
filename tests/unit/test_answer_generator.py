import pytest

from config import prompts_config
from generation.answer_generator import AnswerGenerator
from utils.models import Chunk, RetrievedChunk


def test_answer_generator_citations_formatted(mock_llm_call):
    generator = AnswerGenerator()

    c1 = Chunk(text="RAG improves context.", source="doc1.txt", chunk_id="1", page=2)
    c2 = Chunk(
        text="Vector search retrieves chunks.", source="doc2.txt", chunk_id="2", page=5
    )
    retrieved = [
        RetrievedChunk(chunk=c1, score=0.9),
        RetrievedChunk(chunk=c2, score=0.8),
    ]

    response = generator.generate(query="What is RAG?", retrieved_chunks=retrieved)

    assert response.is_fallback is False
    assert (
        response.answer
        == "This is a test generated answer based on retrieved context [1]."
    )
    assert len(response.citations) == 2
    assert response.citations[0] == "[1] Source: doc1.txt, Page: 2"
    assert response.citations[1] == "[2] Source: doc2.txt, Page: 5"
    assert mock_llm_call.called


def test_answer_generator_empty_chunks_triggers_fallback(mock_llm_call):
    generator = AnswerGenerator()

    response = generator.generate(query="Unanswerable question?", retrieved_chunks=[])

    assert response.is_fallback is True
    assert response.answer == prompts_config.fallback_response
    assert response.citations == []
    mock_llm_call.assert_not_called()


def test_answer_generator_llm_explicit_fallback_signal(mock_llm_call):
    mock_llm_call.return_value = prompts_config.fallback_response
    generator = AnswerGenerator()

    c1 = Chunk(text="Some text", source="doc.txt", chunk_id="1", page=1)
    retrieved = [RetrievedChunk(chunk=c1, score=0.9)]

    response = generator.generate(
        query="Out of scope query?", retrieved_chunks=retrieved
    )
    assert response.is_fallback is True


def test_answer_generator_provider_init_validation(monkeypatch, mocker):
    mocker.patch("anthropic.Anthropic")
    mocker.patch("openai.OpenAI")

    # Clear all keys
    monkeypatch.setattr("generation.answer_generator.ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("generation.answer_generator.OPENAI_API_KEY", "")
    monkeypatch.setattr("generation.answer_generator.DEEPSEEK_API_KEY", "")
    monkeypatch.setattr("generation.answer_generator.OPENROUTER_API_KEY", "")
    monkeypatch.setattr("generation.answer_generator.GEMINI_API_KEY", "")

    # Invalid provider
    with pytest.raises(ValueError):
        AnswerGenerator(provider="invalid_provider")

    # Missing API keys raises EnvironmentError
    with pytest.raises(EnvironmentError):
        AnswerGenerator(provider="anthropic")

    with pytest.raises(EnvironmentError):
        AnswerGenerator(provider="openai")

    with pytest.raises(EnvironmentError):
        AnswerGenerator(provider="deepseek")

    with pytest.raises(EnvironmentError):
        AnswerGenerator(provider="openrouter")

    with pytest.raises(EnvironmentError):
        AnswerGenerator(provider="gemini")

    # Valid API keys
    monkeypatch.setattr("generation.answer_generator.ANTHROPIC_API_KEY", "dummy-key")
    gen_anthropic = AnswerGenerator(provider="anthropic")
    assert gen_anthropic.provider == "anthropic"

    monkeypatch.setattr("generation.answer_generator.OPENAI_API_KEY", "dummy-key")
    gen_openai = AnswerGenerator(provider="openai")
    assert gen_openai.provider == "openai"

    monkeypatch.setattr("generation.answer_generator.DEEPSEEK_API_KEY", "dummy-key")
    gen_deepseek = AnswerGenerator(provider="deepseek")
    assert gen_deepseek.provider == "deepseek"

    monkeypatch.setattr("generation.answer_generator.OPENROUTER_API_KEY", "dummy-key")
    gen_openrouter = AnswerGenerator(provider="openrouter")
    assert gen_openrouter.provider == "openrouter"

    monkeypatch.setattr("generation.answer_generator.GEMINI_API_KEY", "dummy-key")
    gen_gemini = AnswerGenerator(provider="gemini")
    assert gen_gemini.provider == "gemini"


def test_call_anthropic_and_openai_methods(monkeypatch, mocker):
    monkeypatch.setattr("generation.answer_generator.ANTHROPIC_API_KEY", "dummy-key")
    monkeypatch.setattr("generation.answer_generator.OPENAI_API_KEY", "dummy-key")

    # Mock anthropic client call
    mock_anthropic_client = mocker.MagicMock()
    mock_msg = mocker.MagicMock()
    mock_msg.content = [mocker.MagicMock(text="Claude response")]
    mock_anthropic_client.messages.create.return_value = mock_msg
    mocker.patch("anthropic.Anthropic", return_value=mock_anthropic_client)

    gen_a = AnswerGenerator(provider="anthropic")
    assert gen_a._call_llm("prompt") == "Claude response"

    # Mock openai client call
    mock_openai_client = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = "OpenAI response"
    mock_openai_client.chat.completions.create.return_value.choices = [mock_choice]
    mocker.patch("openai.OpenAI", return_value=mock_openai_client)

    gen_o = AnswerGenerator(provider="openai")
    assert gen_o._call_llm("prompt") == "OpenAI response"
