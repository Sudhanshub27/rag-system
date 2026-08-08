"""
LLM Answer Generator
Formats retrieved context into a prompt and calls the configured LLM.
Enforces citation-grounded answers and detects insufficient-context situations.
"""

import sys

from config import (
    generation_config,
    get_api_key,
    prompts_config,
)
from utils.helpers import format_citations
from utils.logger import logger
from utils.models import RAGResponse, RetrievedChunk

# Module-level API key attributes (supports pytest monkeypatching & st.secrets)
ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
DEEPSEEK_API_KEY = get_api_key("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")
GEMINI_API_KEY = get_api_key("GEMINI_API_KEY")


class AnswerGenerator:
    """
    Generate answers strictly grounded in retrieved chunks.

    Supports:
      - Anthropic Claude
      - OpenAI GPT models
      - DeepSeek
      - OpenRouter
      - Google Gemini

    Args:
        provider:    'anthropic', 'openai', 'deepseek', 'openrouter', or 'gemini'.
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
        retrieved_chunks: list[RetrievedChunk],
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
        cleaned_answer = self._clean_reasoning(raw_answer)

        # Check for explicit fallback signal from LLM
        is_fallback = prompts_config.fallback_response.lower() in cleaned_answer.lower()

        citations = format_citations(retrieved_chunks)

        # Self-RAG ML Evaluation: Faithfulness & Context Relevance
        faithfulness, relevance = self.evaluate_faithfulness_and_relevance(
            cleaned_answer, query, retrieved_chunks
        )

        logger.info(
            f"Answer generated. is_fallback={is_fallback}, "
            f"faithfulness={faithfulness}, relevance={relevance}, "
            f"len={len(cleaned_answer)} chars"
        )

        return RAGResponse(
            answer=cleaned_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            query=query,
            is_fallback=is_fallback,
            faithfulness_score=faithfulness,
            relevance_score=relevance,
        )

    def generate_hyde_doc(self, query: str) -> str:
        """
        Generate a hypothetical document passage for HyDE (Hypothetical Document Embeddings) retrieval.
        """
        prompt = (
            f"Write a short, realistic 2-3 sentence passage that directly answers this question:\n"
            f"Question: {query}\n\nPassage:"
        )
        try:
            raw = self._call_llm(prompt)
            return self._clean_reasoning(raw)
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query

    def generate_query_expansions(self, query: str, num_queries: int = 2) -> list[str]:
        """
        Generate semantic variations of the user query for Multi-Query Expansion retrieval.
        """
        prompt = (
            f"Generate {num_queries} alternative search queries for searching a document database.\n"
            f"Output ONLY the queries, one per line. No numbers or prefixes.\n\nQuery: {query}"
        )
        try:
            raw = self._call_llm(prompt)
            cleaned = self._clean_reasoning(raw)
            queries = [
                q.strip("- 123456789.") for q in cleaned.splitlines() if q.strip()
            ]
            return queries[:num_queries]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []

    def evaluate_faithfulness_and_relevance(
        self, answer: str, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[float, float]:
        """
        Compute quantitative Faithfulness and Context Relevance scores (Self-RAG evaluation).
        """
        if not answer or not chunks:
            return 0.0, 0.0

        import re

        context_text = " ".join(rc.chunk.text.lower() for rc in chunks)

        # Faithfulness: answer words present in context
        words = [
            w.lower()
            for w in re.findall(r"\w{4,}", answer)
            if w.lower()
            not in {
                "this",
                "that",
                "with",
                "from",
                "have",
                "been",
                "were",
                "source",
                "page",
            }
        ]
        if not words:
            faithfulness = 1.0
        else:
            matches = sum(1 for w in words if w in context_text)
            faithfulness = round(matches / len(words), 2)

        # Context Relevance: query words present in context
        q_words = [
            w.lower()
            for w in re.findall(r"\w{3,}", query)
            if w.lower()
            not in {
                "what",
                "where",
                "when",
                "show",
                "give",
                "list",
                "tell",
                "from",
                "with",
            }
        ]
        if not q_words:
            relevance = 1.0
        else:
            rel_matches = sum(1 for w in q_words if w in context_text)
            relevance = round(min(1.0, rel_matches / len(q_words)), 2)

        return min(1.0, max(0.0, faithfulness)), min(1.0, max(0.0, relevance))

    @staticmethod
    def _clean_reasoning(text: str) -> str:
        """Strip internal reasoning traces, <think> tags, and meta preamble from model output."""
        import re

        if not text:
            return ""

        # 1. Strip <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # 2. Strip safety guardrail metadata lines (e.g., "User Safety: safe")
        text = re.sub(
            r"^(user\s+safety:\s*safe\s*)+", "", text, flags=re.IGNORECASE
        ).strip()
        text = re.sub(
            r"\n(user\s+safety:\s*safe\s*)+", "\n", text, flags=re.IGNORECASE
        ).strip()

        # 3. Check if text starts with thinking preamble before "Answer:" or "Final Answer:"
        for marker in ("\nAnswer:", "\nFinal Answer:", "Answer:"):
            if marker in text:
                parts = text.split(marker, 1)
                preamble = parts[0].strip().lower()
                if (
                    any(
                        k in preamble
                        for k in (
                            "we need to",
                            "thinking",
                            "reasoning",
                            "context chunks",
                            "instructions:",
                            "let's analyze",
                            "user question",
                        )
                    )
                    or len(preamble) > 30
                ):
                    text = parts[1]
                    break

        # 4. Clean leading 'Answer:' or 'Final Answer:' labels
        text = re.sub(
            r"^(Answer|Final Answer):\s*", "", text.strip(), flags=re.IGNORECASE
        )

        # 5. Remove leftover leading preamble lines
        lines = text.splitlines()
        filtered_lines = []
        skipping_preamble = True
        for line in lines:
            stripped = line.strip()
            if skipping_preamble:
                if any(
                    stripped.lower().startswith(p)
                    for p in (
                        "we need to answer",
                        "thinking process",
                        "reasoning process",
                        "the context chunks are",
                        "we must cite",
                    )
                ):
                    continue
                if stripped:
                    skipping_preamble = False
            filtered_lines.append(line)

        result = "\n".join(filtered_lines).strip()
        return result if result else text.strip()

    # ── Context builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
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

    @staticmethod
    def _get_key(key_name: str) -> str:
        """Retrieve API key from module attribute (if monkeypatched), env, or Streamlit secrets."""
        mod_val = getattr(sys.modules[__name__], key_name, None)
        if mod_val is not None:
            return str(mod_val).strip()
        return get_api_key(key_name)

    def _init_client(self):
        """Initialize the appropriate LLM client with auto-detection if configured key is missing."""
        provider = self.provider

        # Auto-detect active provider if requested provider's key is not set
        if provider == "anthropic" and not self._get_key("ANTHROPIC_API_KEY"):
            provider = self._auto_detect_provider()
        elif provider == "openai" and not self._get_key("OPENAI_API_KEY"):
            provider = self._auto_detect_provider()
        elif provider == "deepseek" and not self._get_key("DEEPSEEK_API_KEY"):
            provider = self._auto_detect_provider()
        elif provider == "openrouter" and not self._get_key("OPENROUTER_API_KEY"):
            provider = self._auto_detect_provider()
        elif provider == "gemini" and not self._get_key("GEMINI_API_KEY"):
            provider = self._auto_detect_provider()

        self.provider = provider

        if self.provider == "anthropic":
            key = self._get_key("ANTHROPIC_API_KEY")
            if not key:
                raise OSError("ANTHROPIC_API_KEY environment variable is not set.")
            try:
                import anthropic

                return anthropic.Anthropic(api_key=key)
            except ImportError as e:
                raise ImportError(
                    "anthropic package not found. Run: pip install anthropic"
                ) from e

        elif self.provider == "openai":
            key = self._get_key("OPENAI_API_KEY")
            if not key:
                raise OSError("OPENAI_API_KEY environment variable is not set.")
            try:
                from openai import OpenAI

                return OpenAI(api_key=key)
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "deepseek":
            key = self._get_key("DEEPSEEK_API_KEY")
            if not key:
                raise OSError(
                    "DEEPSEEK_API_KEY environment variable is not set. "
                    "Get a key at https://platform.deepseek.com"
                )
            try:
                from openai import OpenAI

                if self.model == generation_config.model or "free" in self.model:
                    self.model = "deepseek-chat"

                return OpenAI(
                    api_key=key,
                    base_url="https://api.deepseek.com",
                )
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "openrouter":
            key = self._get_key("OPENROUTER_API_KEY")
            if not key:
                raise OSError(
                    "OPENROUTER_API_KEY environment variable is not set. "
                    "Get a key at https://openrouter.ai/keys"
                )
            try:
                from openai import OpenAI

                return OpenAI(
                    api_key=key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/Sudhanshub27/rag-system",
                        "X-Title": "Ask My Documents RAG",
                    },
                )
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "gemini":
            key = self._get_key("GEMINI_API_KEY")
            if not key:
                raise OSError(
                    "GEMINI_API_KEY environment variable is not set. "
                    "Get a key at https://aistudio.google.com/app/apikey"
                )
            try:
                from openai import OpenAI

                if self.model == generation_config.model or "free" in self.model:
                    self.model = "gemini-2.0-flash"

                return OpenAI(
                    api_key=key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        else:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                "Use 'anthropic', 'openai', 'deepseek', 'openrouter', or 'gemini'."
            )

    @classmethod
    def _auto_detect_provider(cls) -> str:
        """Find the first available API key in environment variables or Streamlit secrets."""
        if cls._get_key("DEEPSEEK_API_KEY"):
            logger.info("Auto-detected DEEPSEEK_API_KEY in environment/secrets")
            return "deepseek"
        if cls._get_key("OPENROUTER_API_KEY"):
            logger.info("Auto-detected OPENROUTER_API_KEY in environment/secrets")
            return "openrouter"
        if cls._get_key("GEMINI_API_KEY"):
            logger.info("Auto-detected GEMINI_API_KEY in environment/secrets")
            return "gemini"
        if cls._get_key("ANTHROPIC_API_KEY"):
            logger.info("Auto-detected ANTHROPIC_API_KEY in environment/secrets")
            return "anthropic"
        if cls._get_key("OPENAI_API_KEY"):
            logger.info("Auto-detected OPENAI_API_KEY in environment/secrets")
            return "openai"
        return "openrouter"

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to LLM and return response text."""
        system_prompt = prompts_config.system_prompt

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, prompt)
            elif self.provider in ("openai", "deepseek", "openrouter", "gemini"):
                # All four use the OpenAI-compatible client
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
