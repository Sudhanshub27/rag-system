"""
LLM Answer Generator
Formats retrieved context into a prompt and calls the configured LLM.
Supports Groq, Ollama (Best Local Models), Anthropic, OpenAI, DeepSeek, OpenRouter, and Gemini.
Enforces citation-grounded answers, PII anonymization, and detects insufficient-context situations.
"""

import json
import sys
import urllib.request

from config import (
    generation_config,
    get_api_key,
    prompts_config,
)
from utils.anonymizer import pii_anonymizer
from utils.helpers import format_citations
from utils.logger import logger
from utils.models import RAGResponse, RetrievedChunk

# Module-level API key attributes (supports pytest monkeypatching & st.secrets)
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
DEEPSEEK_API_KEY = get_api_key("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")
GEMINI_API_KEY = get_api_key("GEMINI_API_KEY")


class AnswerGenerator:
    """
    Generate answers strictly grounded in retrieved chunks.

    Supports:
      - Groq (Llama-3.3-70b, Llama-3.1-8b - Free & Zero Training)
      - Ollama (Best Local Models: Llama 3.3, Qwen 2.5, Mistral)
      - Anthropic Claude
      - OpenAI GPT models
      - DeepSeek
      - OpenRouter
      - Google Gemini

    Args:
        provider:    'groq', 'ollama', 'anthropic', 'openai', 'deepseek', 'openrouter', or 'gemini'.
        model:       Model name string.
        max_tokens:  Maximum tokens in the generated response.
        temperature: Sampling temperature (low = more deterministic).
        api_key:     Optional custom API key (Bring Your Own API Key - BYOK).
    """

    def __init__(
        self,
        provider: str = generation_config.provider,
        model: str = generation_config.model,
        max_tokens: int = generation_config.max_tokens,
        temperature: float = generation_config.temperature,
        api_key: str | None = None,
    ):
        self.custom_api_key = api_key
        self.provider = (provider or "groq").lower()
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
        anonymize_pii: bool = False,
    ) -> RAGResponse:
        """
        Generate a cited answer from retrieved chunks.

        Args:
            query:               The user's question.
            retrieved_chunks:    Chunks returned by the retriever.
            min_chunks_required: If fewer chunks are available, return fallback.
            anonymize_pii:       Redact names, emails, phones before sending to LLM.

        Returns:
            RAGResponse with answer text, citations, and metadata.
        """
        logger.info(
            f"Generating answer for query: '{query[:80]}' (provider={self.provider}, anonymize_pii={anonymize_pii})"
        )
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
        pii_mapping = {}

        # Local PII Anonymization Layer (Redact PII before sending to external API)
        if anonymize_pii:
            query_anon, q_map = pii_anonymizer.anonymize(query)
            context_anon, c_map = pii_anonymizer.anonymize(context)
            query = query_anon
            context = context_anon
            pii_mapping = {**q_map, **c_map}

        # Build the full prompt
        prompt = prompts_config.answer_prompt.format(
            context=context,
            question=query,
        )

        # Call the LLM with safe fallback exception handling
        try:
            raw_answer = self._call_llm(prompt)
            cleaned_answer = self._clean_reasoning(raw_answer)
        except Exception as e:
            logger.error(f"LLM execution error: {e}")
            return RAGResponse(
                answer=f"⚠️ Unable to generate response with provider '{self.provider}'. {e}\n\nFallback: {prompts_config.fallback_response}",
                citations=[],
                retrieved_chunks=retrieved_chunks,
                query=query,
                is_fallback=True,
            )

        # De-anonymize PII placeholders back to real tokens if anonymization was enabled
        if anonymize_pii and pii_mapping:
            cleaned_answer = pii_anonymizer.deanonymize(cleaned_answer, pii_mapping)

        # Check for explicit fallback signals or empty answers
        fallback_signals = [
            prompts_config.fallback_response.lower(),
            "could not find relevant information in your uploaded documents",
            "no relevant information found in the uploaded documents",
            "i could not find any information regarding",
            "unable to find relevant information",
        ]
        lower_ans = cleaned_answer.lower().strip()
        is_fallback = (
            any(sig in lower_ans for sig in fallback_signals)
            or (lower_ans.startswith("i could not find") and len(lower_ans) < 140)
            or len(cleaned_answer.strip()) < 5
        )

        if is_fallback:
            cleaned_answer = prompts_config.fallback_response
            citations = []
            retrieved_chunks = []
        else:
            citations = format_citations(retrieved_chunks)

        # Self-RAG ML Evaluation: Faithfulness & Context Relevance
        faithfulness, relevance = (
            (0.0, 0.0)
            if is_fallback
            else self.evaluate_faithfulness_and_relevance(
                cleaned_answer, query, retrieved_chunks
            )
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

    def generate_summary(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        anonymize_pii: bool = False,
    ) -> RAGResponse:
        """
        Generate a section-structured document summary from ordered document chunks.
        """
        logger.info(f"Generating summary for query: '{query[:80]}'")
        if not retrieved_chunks:
            return RAGResponse(
                answer=prompts_config.fallback_response,
                citations=[],
                retrieved_chunks=[],
                query=query,
                is_fallback=True,
            )

        context_parts = []
        for rc in retrieved_chunks:
            c = rc.chunk
            pg = c.page
            c_idx = c.metadata.get("chunk_index", 0)
            context_parts.append(f"--- [Page {pg} | Chunk {c_idx}] ---\n{c.text}")
        context = "\n\n".join(context_parts)

        pii_mapping = {}
        if anonymize_pii:
            query, q_map = pii_anonymizer.anonymize(query)
            context, c_map = pii_anonymizer.anonymize(context)
            pii_mapping = {**q_map, **c_map}

        summarize_template = getattr(
            prompts_config,
            "summarize_prompt",
            prompts_config.answer_prompt,
        )

        prompt = summarize_template.format(
            context=context,
            question=query,
        )

        raw_answer = self._call_llm(prompt)
        cleaned_answer = self._clean_reasoning(raw_answer)

        if anonymize_pii and pii_mapping:
            cleaned_answer = pii_anonymizer.deanonymize(cleaned_answer, pii_mapping)

        is_fallback = prompts_config.fallback_response.lower() in cleaned_answer.lower()
        citations = format_citations(retrieved_chunks)

        return RAGResponse(
            answer=cleaned_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            query=query,
            is_fallback=is_fallback,
            faithfulness_score=1.0 if not is_fallback else 0.0,
            relevance_score=1.0 if not is_fallback else 0.0,
        )

    def generate_hyde_doc(self, query: str) -> str:
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
        if not answer or not chunks:
            return 0.0, 0.0

        import re

        context_text = " ".join(rc.chunk.text.lower() for rc in chunks)

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

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(
            r"^(user\s+safety:\s*safe\s*)+", "", text, flags=re.IGNORECASE
        ).strip()
        text = re.sub(
            r"\n(user\s+safety:\s*safe\s*)+", "\n", text, flags=re.IGNORECASE
        ).strip()

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

        text = re.sub(
            r"^(Answer|Final Answer):\s*", "", text.strip(), flags=re.IGNORECASE
        )

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

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, rc in enumerate(chunks, start=1):
            header = f"[{i}] Source: {rc.chunk.source}, Page: {rc.chunk.page}"
            parts.append(f"{header}\n{rc.chunk.text}")
        return "\n\n".join(parts)

    @classmethod
    def _get_key(cls, key_name: str, custom_key: str | None = None) -> str:
        """Retrieve API key from custom BYOK argument, module attribute, env, or Streamlit secrets."""
        if custom_key:
            return custom_key.strip()
        mod_val = getattr(sys.modules[__name__], key_name, None)
        if mod_val is not None and str(mod_val).strip():
            return str(mod_val).strip()
        return get_api_key(key_name)

    @classmethod
    def _get_keys_list(cls, key_name: str, custom_key: str | None = None) -> list[str]:
        """Retrieve list of API keys (supports comma-separated string in env/secrets)."""
        raw = cls._get_key(key_name, custom_key)
        if not raw:
            return []
        return [k.strip() for k in raw.split(",") if k.strip()]

    @classmethod
    def get_best_ollama_model(cls) -> str:
        """
        Auto-detect the BEST local Ollama model installed on the system.
        Prioritizes: llama3.3 > llama3.1:70b > llama3.1 > qwen2.5:72b > qwen2.5 > mistral > gemma2.
        """
        preferred_models = [
            "llama3.3:latest",
            "llama3.3",
            "llama3.1:70b",
            "llama3.1:latest",
            "llama3.1",
            "qwen2.5:72b",
            "qwen2.5:latest",
            "qwen2.5",
            "mistral:latest",
            "mistral",
            "gemma2:latest",
            "gemma2",
            "llama3:latest",
            "llama3",
        ]
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
                installed = [m.get("name", "") for m in data.get("models", [])]

            logger.info(f"Ollama local models detected: {installed}")
            for pref in preferred_models:
                for inst in installed:
                    if pref == inst or pref.split(":")[0] in inst:
                        logger.info(f"Selected best Ollama model: '{inst}'")
                        return inst

            if installed:
                return installed[0]
        except Exception as e:
            logger.warning(f"Could not fetch Ollama local tags: {e}")

        return "llama3.3:latest"

    def _init_client(self):
        """Initialize the appropriate LLM client with auto-detection if key is missing."""
        provider = self.provider

        if provider == "groq" and not self._get_key(
            "GROQ_API_KEY", self.custom_api_key
        ):
            provider = self._auto_detect_provider()
        elif provider == "anthropic" and not self._get_key(
            "ANTHROPIC_API_KEY", self.custom_api_key
        ):
            provider = self._auto_detect_provider()
        elif provider == "openai" and not self._get_key(
            "OPENAI_API_KEY", self.custom_api_key
        ):
            provider = self._auto_detect_provider()
        elif provider == "deepseek" and not self._get_key(
            "DEEPSEEK_API_KEY", self.custom_api_key
        ):
            provider = self._auto_detect_provider()
        elif provider == "openrouter" and not self._get_key(
            "OPENROUTER_API_KEY", self.custom_api_key
        ):
            provider = self._auto_detect_provider()
        elif provider == "gemini" and not self._get_key(
            "GEMINI_API_KEY", self.custom_api_key
        ):
            provider = self._auto_detect_provider()

        self.provider = provider

        if self.provider == "groq":
            key = self._get_key("GROQ_API_KEY", self.custom_api_key)
            if not key:
                raise OSError(
                    "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys"
                )
            try:
                from openai import OpenAI

                if (
                    not self.model
                    or self.model == generation_config.model
                    or "decommissioned" in self.model
                ):
                    self.model = "openai/gpt-oss-120b"
                return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "ollama":
            best_model = self.get_best_ollama_model()
            if not self.model or self.model == generation_config.model:
                self.model = best_model
            logger.info(
                f"Initializing Ollama client with model='{self.model}' at http://localhost:11434/v1"
            )
            try:
                from openai import OpenAI

                return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "anthropic":
            key = self._get_key("ANTHROPIC_API_KEY", self.custom_api_key)
            if not key:
                raise OSError("ANTHROPIC_API_KEY environment variable is not set.")
            try:
                import anthropic

                if not self.model or self.model == generation_config.model:
                    self.model = "claude-3-5-sonnet-20241022"
                return anthropic.Anthropic(api_key=key)
            except ImportError as e:
                raise ImportError(
                    "anthropic package not found. Run: pip install anthropic"
                ) from e

        elif self.provider == "openai":
            key = self._get_key("OPENAI_API_KEY", self.custom_api_key)
            if not key:
                raise OSError("OPENAI_API_KEY environment variable is not set.")
            try:
                from openai import OpenAI

                if not self.model or self.model == generation_config.model:
                    self.model = "gpt-4o-mini"
                return OpenAI(api_key=key)
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "deepseek":
            key = self._get_key("DEEPSEEK_API_KEY", self.custom_api_key)
            if not key:
                raise OSError("DEEPSEEK_API_KEY environment variable is not set.")
            try:
                from openai import OpenAI

                if not self.model or self.model == generation_config.model:
                    self.model = "deepseek-chat"
                return OpenAI(api_key=key, base_url="https://api.deepseek.com")
            except ImportError as e:
                raise ImportError(
                    "openai package not found. Run: pip install openai"
                ) from e

        elif self.provider == "openrouter":
            key = self._get_key("OPENROUTER_API_KEY", self.custom_api_key)
            if not key:
                raise OSError("OPENROUTER_API_KEY environment variable is not set.")
            try:
                from openai import OpenAI

                if not self.model or self.model == generation_config.model:
                    self.model = "meta-llama/llama-3.3-70b-instruct"
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
            key = self._get_key("GEMINI_API_KEY", self.custom_api_key)
            if not key:
                raise OSError("GEMINI_API_KEY environment variable is not set.")
            try:
                from openai import OpenAI

                if not self.model or self.model == generation_config.model:
                    self.model = "gemini-1.5-flash"
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
                "Use 'groq', 'ollama', 'anthropic', 'openai', 'deepseek', 'openrouter', or 'gemini'."
            )

    @classmethod
    def _auto_detect_provider(cls) -> str:
        """Find the first available API key in environment variables or Streamlit secrets."""
        if cls._get_key("GROQ_API_KEY"):
            logger.info("Auto-detected GROQ_API_KEY in environment/secrets")
            return "groq"

        # Check if Ollama local server is running
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    logger.info("Auto-detected local Ollama server running")
                    return "ollama"
        except Exception:
            pass

        if cls._get_key("OPENAI_API_KEY"):
            logger.info("Auto-detected OPENAI_API_KEY in environment/secrets")
            return "openai"
        if cls._get_key("ANTHROPIC_API_KEY"):
            logger.info("Auto-detected ANTHROPIC_API_KEY in environment/secrets")
            return "anthropic"
        if cls._get_key("DEEPSEEK_API_KEY"):
            logger.info("Auto-detected DEEPSEEK_API_KEY in environment/secrets")
            return "deepseek"
        return "groq"

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to LLM and return response text."""
        system_prompt = prompts_config.system_prompt

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, prompt)
            elif self.provider in (
                "groq",
                "ollama",
                "openai",
                "deepseek",
                "openrouter",
                "gemini",
            ):
                return self._call_openai(system_prompt, prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def _call_anthropic(self, system: str, user: str) -> str:
        models_to_try = [self.model]
        fallback_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
        ]
        for fb in fallback_models:
            if fb not in models_to_try:
                models_to_try.append(fb)

        keys = self._get_keys_list("ANTHROPIC_API_KEY", self.custom_api_key)
        if not keys:
            keys = [getattr(self._client, "api_key", "")]

        import anthropic

        last_error = None
        for m in models_to_try:
            for key in keys:
                if not key:
                    continue
                client = anthropic.Anthropic(api_key=key)
                try:
                    response = client.messages.create(
                        model=m,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    if m != self.model or key != keys[0]:
                        logger.info(
                            f"Anthropic execution succeeded with model '{m}' (key prefix '{key[:8]}...')"
                        )
                    return response.content[0].text.strip()
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Anthropic key '{key[:8]}...' model '{m}' call failed: {e}. Trying next key/model..."
                    )

        raise last_error

    def _call_openai(self, system: str, user: str) -> str:
        models_to_try = [self.model]
        if self.provider == "groq":
            fallback_models = [
                "openai/gpt-oss-120b",
                "groq/compound",
                "qwen/qwen3.6-27b",
                "groq/compound-mini",
            ]
        elif self.provider == "openai":
            fallback_models = [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "o3-mini",
                "o1-mini",
            ]
        elif self.provider == "deepseek":
            fallback_models = [
                "deepseek-chat",
                "deepseek-reasoner",
            ]
        elif self.provider == "gemini":
            fallback_models = [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-2.0-flash",
                "gemini-1.0-pro",
            ]
        elif self.provider == "openrouter":
            fallback_models = [
                "meta-llama/llama-3.3-70b-instruct",
                "anthropic/claude-3.5-sonnet",
                "deepseek/deepseek-r1",
                "openai/gpt-4o",
            ]
        else:
            fallback_models = []

        for fb in fallback_models:
            if fb not in models_to_try:
                models_to_try.append(fb)

        # Retrieve key list (supports comma-separated keys in env/secrets)
        keys = []
        if self.provider == "groq":
            keys = self._get_keys_list("GROQ_API_KEY", self.custom_api_key)
        elif self.provider == "gemini":
            keys = self._get_keys_list("GEMINI_API_KEY", self.custom_api_key)
        elif self.provider == "openrouter":
            keys = self._get_keys_list("OPENROUTER_API_KEY", self.custom_api_key)
        elif self.provider == "openai":
            keys = self._get_keys_list("OPENAI_API_KEY", self.custom_api_key)
        elif self.provider == "deepseek":
            keys = self._get_keys_list("DEEPSEEK_API_KEY", self.custom_api_key)

        if not keys:
            keys = [getattr(self._client, "api_key", "")]

        last_error = None
        from openai import OpenAI

        base_url = getattr(self._client, "base_url", None)
        default_headers = getattr(self._client, "default_headers", None)

        for m in models_to_try:
            for key in keys:
                if not key:
                    continue
                client = OpenAI(
                    api_key=key,
                    base_url=base_url,
                    default_headers=default_headers,
                )
                try:
                    response = client.chat.completions.create(
                        model=m,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    if m != self.model or key != keys[0]:
                        logger.info(
                            f"LLM execution succeeded with model '{m}' (key prefix '{key[:8]}...')"
                        )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Key '{key[:8]}...' model '{m}' call failed: {e}. Trying next key/model..."
                    )

        raise last_error
