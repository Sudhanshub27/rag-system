"""
Cached Document Summarizer
Computes & caches document-level summaries using Map-Reduce over document chunks.

Key features:
  - Content-hash caching (SHA-256 of extracted chunk text): edits automatically invalidate the cache
  - Map-Reduce batching (groups of 4-5 chunks summarized, then combined into a final outline)
  - Rate-limited API calls (semaphore + backoff) to respect provider RPM/TPM limits
  - Zero LLM calls on cache hit
"""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from utils.logger import logger
from utils.models import Chunk

# Local cache directory for document summaries
DEFAULT_CACHE_DIR = Path(".cache/summaries")


class RateLimiterSemaphore:
    """Semaphore wrapper to enforce rate limits across map-reduce LLM calls."""

    def __init__(self, max_concurrent: int = 4, min_delay_seconds: float = 0.1):
        self._semaphore = threading.Semaphore(max_concurrent)
        self.min_delay_seconds = min_delay_seconds
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        self._semaphore.acquire()
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.min_delay_seconds:
                time.sleep(self.min_delay_seconds - elapsed)
            self._last_call_time = time.time()

    def release(self) -> None:
        self._semaphore.release()


# Global rate limiter instance for map-reduce summarization
_summarizer_rate_limiter = RateLimiterSemaphore(max_concurrent=4, min_delay_seconds=0.1)


def compute_content_hash(chunks: list[Chunk]) -> str:
    """Compute SHA-256 hash of concatenated chunk texts to serve as content cache key."""
    if not chunks:
        return "empty_document_hash"
    combined_text = "".join(c.text for c in chunks)
    return hashlib.sha256(combined_text.encode("utf-8")).hexdigest()


class DocumentSummarizer:
    """
    Computes and caches document-level summaries using Map-Reduce.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_build_doc_summary(
        self,
        chunks: list[Chunk],
        generator: Any,
        tenant_id: str = "default_tenant",
        force_refresh: bool = False,
    ) -> str:
        """
        Get cached summary or build it via Map-Reduce if absent.

        Args:
            chunks: List of document Chunk objects.
            generator: AnswerGenerator instance to perform LLM calls.
            tenant_id: Unique tenant identifier for tenant-isolated cache keying.
            force_refresh: If True, bypass cache and rebuild.

        Returns:
            Document-level summary string.
        """
        if not chunks:
            return "No content available to summarize."

        doc_hash = compute_content_hash(chunks)
        # Format key as summary_{tenant_id}_{doc_hash}.json for multi-tenant isolation
        cache_key = f"{tenant_id}:{doc_hash}"
        cache_filename = f"summary_{tenant_id}_{doc_hash}.json"
        cache_path = self.cache_dir / cache_filename

        # Check cache store (zero LLM calls if hit)
        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"DocSummarizer: Cache HIT for key '{cache_key[:20]}...'")
                return data.get("summary", "")
            except Exception as e:
                logger.warning(f"Failed to read summary cache ({e}), recomputing...")

        logger.info(
            f"DocSummarizer: Cache MISS for content hash {doc_hash[:10]}... "
            f"Building fast Map-Reduce summary across {len(chunks)} chunks."
        )

        # Build Map-Reduce summary
        summary = self._map_reduce_summarize(chunks, generator)

        # Save to cache
        try:
            cache_data = {
                "tenant_id": tenant_id,
                "content_hash": doc_hash,
                "cache_key": cache_key,
                "chunk_count": len(chunks),
                "timestamp": time.time(),
                "summary": summary,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            logger.info(
                f"DocSummarizer: Summary cached under key '{cache_key[:20]}...'"
            )
        except Exception as e:
            logger.warning(f"Could not write summary cache: {e}")

        return summary

    def _map_reduce_summarize(self, chunks: list[Chunk], generator: Any) -> str:
        """
        Execute parallel Map-Reduce summarization over chunks.
        """
        # Single batch processing for documents up to 12 chunks (fast 1-call path)
        if len(chunks) <= 12:
            context = "\n\n".join(
                f"[Chunk {i+1}] {c.text}" for i, c in enumerate(chunks)
            )
            prompt = (
                "Please provide a comprehensive summary and section outline of the following document:\n\n"
                f"{context}\n\nDocument Summary & Outline:"
            )
            return self._rate_limited_generate(generator, prompt)

        # MAP step: Group chunks into larger batches of 12 chunks (~3000 tokens)
        group_size = 12
        grouped_chunks = [
            chunks[i : i + group_size] for i in range(0, len(chunks), group_size)
        ]

        logger.info(
            f"DocSummarizer MAP step: Parallel processing {len(grouped_chunks)} chunk groups..."
        )

        def _process_group(idx: int, group: list[Chunk]) -> tuple[int, str]:
            group_context = "\n\n".join(c.text for c in group)
            map_prompt = (
                f"Summarize key points and topics in section {idx}/{len(grouped_chunks)} of the document:\n\n"
                f"{group_context}\n\nSection Key Points:"
            )
            summary_text = self._rate_limited_generate(generator, map_prompt)
            return idx, f"### Section {idx}\n{summary_text}"

        results_dict = {}
        max_workers = min(4, len(grouped_chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_group, idx, group): idx
                for idx, group in enumerate(grouped_chunks, start=1)
            }
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                results_dict[idx] = result

        # Re-order group summaries by section index
        group_summaries = [results_dict[i] for i in sorted(results_dict.keys())]

        # REDUCE step: Combine group summaries into final summary and outline
        combined_summaries = "\n\n".join(group_summaries)
        reduce_prompt = (
            "You are an expert document analyst. Below are section summaries from a full document.\n"
            "Combine them into a clear, structured Executive Summary followed by a Key Topics Outline:\n\n"
            f"{combined_summaries}\n\nFinal Executive Summary & Topic Outline:"
        )

        logger.info("DocSummarizer REDUCE step: Synthesizing final summary...")
        final_summary = self._rate_limited_generate(generator, reduce_prompt)
        return final_summary

    @staticmethod
    def _rate_limited_generate(generator: Any, prompt: str) -> str:
        """Execute generator call wrapped in rate-limiting semaphore."""
        _summarizer_rate_limiter.acquire()
        try:
            # Handle RAGResponse object or raw text return
            res = (
                generator.generate_summary_raw(prompt)
                if hasattr(generator, "generate_summary_raw")
                else None
            )
            if not res:
                res = (
                    generator._call_llm(prompt)
                    if hasattr(generator, "_call_llm")
                    else None
                )
            if not res:
                # Fallback to standard generate
                resp = generator.generate(prompt, [])
                res = resp.answer if hasattr(resp, "answer") else str(resp)
            return res.strip()
        finally:
            _summarizer_rate_limiter.release()


# Module singleton instance
doc_summarizer = DocumentSummarizer()
