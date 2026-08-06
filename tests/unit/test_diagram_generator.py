from generation.diagram_generator import (
    DiagramGenerator,
    _clean_mermaid_output,
    detect_diagram_type,
)
from utils.models import Chunk, RetrievedChunk


def test_detect_diagram_type():
    assert detect_diagram_type("draw a flowchart of the system") == "flowchart TD"
    assert detect_diagram_type("create a class diagram") == "classDiagram"
    assert detect_diagram_type("sequence diagram of login") == "sequenceDiagram"
    assert detect_diagram_type("er diagram for database") == "erDiagram"
    assert detect_diagram_type("mindmap of concepts") == "flowchart LR"
    assert detect_diagram_type("what is RAG?") is None


def test_clean_mermaid_output():
    raw_fenced = "```mermaid\nflowchart TD\n  A --> B\n```"
    cleaned = _clean_mermaid_output(raw_fenced)
    assert cleaned == "flowchart TD\n  A --> B"

    raw_preamble = "Here is the diagram:\nflowchart TD\n  A --> B\n\nExplanation of nodes follows..."
    cleaned_preamble = _clean_mermaid_output(raw_preamble)
    assert "flowchart TD" in cleaned_preamble

    assert _clean_mermaid_output("") == ""


def test_diagram_generator_empty_chunks(mock_llm_call):
    gen = DiagramGenerator()
    resp = gen.generate("flowchart of process", [])
    assert resp.is_fallback is True
    assert "No relevant content" in resp.fallback_message


def test_diagram_generator_success(mock_llm_call, mocker):
    mocker.patch.object(
        DiagramGenerator, "_call_llm", return_value="flowchart TD\n  A --> B"
    )
    gen = DiagramGenerator()

    c = Chunk(text="Step 1 leads to Step 2.", source="doc.txt", chunk_id="c1")
    retrieved = [RetrievedChunk(chunk=c, score=0.9)]

    resp = gen.generate("flowchart of steps", retrieved)
    assert resp.is_fallback is False
    assert resp.mermaid_code == "flowchart TD\n  A --> B"
    assert resp.diagram_type == "flowchart TD"
