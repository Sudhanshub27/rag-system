"""
Diagram Generator
=================
Generates Mermaid diagram syntax from retrieved document context.
Supports: flowchart, classDiagram, sequenceDiagram, erDiagram, mindmap.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import logger
from utils.models import RetrievedChunk

# Keywords that map a user's natural language request to a Mermaid diagram type
DIAGRAM_KEYWORDS: dict[str, list[str]] = {
    "flowchart TD": [
        "flowchart", "flow chart", "flow diagram", "process flow",
        "workflow", "steps", "procedure", "algorithm", "how does",
        "how it works", "dfd", "data flow",
    ],
    "classDiagram": [
        "class diagram", "class structure", "uml class", "classes",
        "inheritance", "object model", "oop", "entities and attributes",
    ],
    "sequenceDiagram": [
        "sequence diagram", "interaction diagram", "message flow",
        "communication", "sequence of", "request response", "api flow",
    ],
    "erDiagram": [
        "er diagram", "entity relationship", "database schema",
        "erd", "tables", "relations", "schema",
    ],
    "mindmap": [
        "mind map", "mindmap", "concept map", "topic map",
        "key concepts", "summarize topics", "overview",
    ],
}


@dataclass
class DiagramResponse:
    """Result of a diagram generation request."""
    mermaid_code: str
    diagram_type: str
    question: str
    source_chunks: List[RetrievedChunk] = field(default_factory=list)
    is_fallback: bool = False
    fallback_message: str = ""


def detect_diagram_type(query: str) -> Optional[str]:
    """
    Detect which Mermaid diagram type the user is requesting.
    Returns the Mermaid diagram type string, or None if not a diagram request.
    """
    q = query.lower()
    for mermaid_type, keywords in DIAGRAM_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return mermaid_type
    return None


def _clean_mermaid_output(raw: str) -> str:
    """Strip markdown code fences and whitespace from LLM output."""
    # Remove ```mermaid ... ``` or ``` ... ``` wrapping
    raw = re.sub(r"^```(?:mermaid)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```\s*$", "", raw.strip())
    return raw.strip()


class DiagramGenerator:
    """
    Generates Mermaid diagrams from retrieved document chunks.
    Reuses the same LLM client as AnswerGenerator.
    """

    def __init__(self):
        from generation.answer_generator import AnswerGenerator
        # Reuse the existing LLM client/config — no duplication
        self._base = AnswerGenerator()
        logger.info("DiagramGenerator initialized (reusing AnswerGenerator client)")

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        diagram_type: Optional[str] = None,
    ) -> DiagramResponse:
        """
        Generate a Mermaid diagram from retrieved chunks.

        Args:
            question: The user's diagram request.
            retrieved_chunks: Relevant chunks from the vector store.
            diagram_type: Mermaid diagram type (auto-detected if None).

        Returns:
            DiagramResponse with mermaid_code ready to render.
        """
        # Auto-detect diagram type if not provided
        if diagram_type is None:
            diagram_type = detect_diagram_type(question) or "flowchart TD"

        logger.info(f"Generating {diagram_type} diagram for: '{question}'")

        # No chunks → fallback
        if not retrieved_chunks:
            return DiagramResponse(
                mermaid_code="",
                diagram_type=diagram_type,
                question=question,
                is_fallback=True,
                fallback_message="No relevant content found in documents to generate diagram.",
            )

        # Build context string
        context_parts = []
        for i, rc in enumerate(retrieved_chunks, start=1):
            context_parts.append(f"[Chunk {i}] {rc.chunk.text[:600]}")
        context = "\n\n".join(context_parts)

        # Build prompt from prompts.yaml
        try:
            from config import prompts_config
            diagram_prompt = prompts_config.diagram_prompt.format(
                diagram_type=diagram_type,
                context=context,
                question=question,
            )
            system_prompt = prompts_config.diagram_system_prompt
        except AttributeError:
            # Fallback inline prompts if YAML doesn't have them yet
            diagram_prompt = (
                f"Based on the context below, generate a Mermaid {diagram_type} diagram.\n"
                f"Output ONLY raw Mermaid syntax. No code fences, no explanations.\n\n"
                f"Context:\n{context}\n\nRequest: {question}\n\nMermaid Diagram:"
            )
            system_prompt = (
                "You are a Mermaid diagram expert. Output ONLY valid Mermaid syntax."
            )

        # Call LLM
        try:
            raw_output = self._call_llm(system_prompt, diagram_prompt)
        except Exception as e:
            logger.error(f"Diagram LLM call failed: {e}")
            return DiagramResponse(
                mermaid_code="",
                diagram_type=diagram_type,
                question=question,
                source_chunks=retrieved_chunks,
                is_fallback=True,
                fallback_message=f"Diagram generation failed: {e}",
            )

        # Clean LLM output
        mermaid_code = _clean_mermaid_output(raw_output)

        # Check if LLM signaled insufficient context
        if "INSUFFICIENT_CONTEXT" in mermaid_code or not mermaid_code:
            return DiagramResponse(
                mermaid_code="",
                diagram_type=diagram_type,
                question=question,
                source_chunks=retrieved_chunks,
                is_fallback=True,
                fallback_message="The documents don't contain enough information to draw this diagram.",
            )

        logger.info(f"Diagram generated ({len(mermaid_code)} chars, type={diagram_type})")
        return DiagramResponse(
            mermaid_code=mermaid_code,
            diagram_type=diagram_type,
            question=question,
            source_chunks=retrieved_chunks,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM via the base AnswerGenerator's client."""
        provider = self._base.provider
        client = self._base._client

        if provider == "anthropic":
            response = client.messages.create(
                model=self._base.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()

        else:  # openai / deepseek / openrouter
            response = client.chat.completions.create(
                model=self._base.model,
                max_tokens=1024,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
