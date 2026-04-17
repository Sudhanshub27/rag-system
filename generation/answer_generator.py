"""
LLM Answer Generator
Formats retrieved context into a prompt and calls the configured LLM.
Enforces citation-grounded answers and detects insufficient-context situations.
"""

import re
from typing import List, Optional

from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, generation_config, prompts_config
from config import DEEPSEEK_API_KEY
from utils.helpers import format_citations
from utils.logger import logger
from utils.models import RAGResponse, RetrievedChunk


class AnswerGenerator:
    """
    Generate answers strictly grounded in retrieved chunks.

    Supports:
      - Anthropic Claude (default)
      - OpenAI GPT models

    Args:
        provider:    'anthropic' or 'openai'.
        model:       Model name string.
        max_tokens:  Maximum tokens in the generated response.
        temperature: Sampling temperature (low = more deterministic).
    """

    def __init__(
        self,
        provider: str = generation_config.provider,
        model: str = generation_config.model,
        max_tokens: int = generation_config.max_tokens,
        temperature: float = generation_config.temperature,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = self._init_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        min_chunks_required: int = 1,
    ) -> RAGResponse:
        """
        Generate a cited answer from retrieved chunks.

        Args:
            query:               The user's question.
            retrieved_chunks:    Chunks returned by the retriever.
            min_chunks_required: If fewer chunks are available, return fallback.

        Returns:
            RAGResponse with answer text, citations, and metadata.
        """
        logger.info(f"Generating answer for query: '{query[:80]}'")
        logger.debug(f"Using {len(retrieved_chunks)} context chunk(s)")

        # Guard: empty retrieval
        if len(retrieved_chunks) < min_chunks_required:
            logger.warning("Insufficient retrieved chunks — returning fallback")
            return RAGResponse(
                answer=prompts_config.fallback_response,
                citations=[],
                retrieved_chunks=retrieved_chunks,
                query=query,
                is_fallback=True,
            )

        # Build numbered context string
        context = self._build_context(retrieved_chunks)

        # Build the full prompt
        prompt = prompts_config.answer_prompt.format(
            context=context,
            question=query,
        )

        # Call the LLM
        raw_answer = self._call_llm(prompt)

        # Check for explicit fallback signal from LLM
        is_fallback = prompts_config.fallback_response.lower() in raw_answer.lower()

        citations = format_citations(retrieved_chunks)

        logger.info(
            f"Answer generated. is_fallback={is_fallback}, "
            f"len={len(raw_answer)} chars"
        )

        return RAGResponse(
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            query=query,
            is_fallback=is_fallback,
        )

    # ── Context builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a numbered context block.

        Example output:
            [1] Source: file.pdf, Page: 3
            <chunk text>

            [2] Source: file.pdf, Page: 7
            <chunk text>
        """
        parts = []
        for i, rc in enumerate(chunks, start=1):
            header = f"[{i}] Source: {rc.chunk.source}, Page: {rc.chunk.page}"
            parts.append(f"{header}\n{rc.chunk.text}")
        return "\n\n".join(parts)

    # ── LLM callers ───────────────────────────────────────────────────────────

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set."
                )
            try:
                import anthropic
                return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError as e:
                raise ImportError(
                    "anthropic package not found. Run: pip install anthropic"
                ) from e

        elif self.provider == "openai":
            if not OPENAI_API_KEY:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set."
                )
            try:
                from openai import OpenAI
                return OpenAI(api_key=OPENAI_API_KEY)
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "deepseek":
            # DeepSeek exposes an OpenAI-compatible REST API
            if not DEEPSEEK_API_KEY:
                raise EnvironmentError(
                    "DEEPSEEK_API_KEY environment variable is not set. "
                    "Get a free key at https://platform.deepseek.com"
                )
            try:
                from openai import OpenAI
                return OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com",
                )
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        else:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                "Use 'anthropic', 'openai', or 'deepseek'."
            )

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to LLM and return response text."""
        system_prompt = prompts_config.system_prompt

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, prompt)
            elif self.provider in ("openai", "deepseek"):
                # Both use the OpenAI-compatible client
                return self._call_openai(system_prompt, prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def _call_anthropic(self, system: str, user: str) -> str:
        """Call Anthropic Claude API."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    def _call_openai(self, system: str, user: str) -> str:
        """Call OpenAI Chat Completions API."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()
