"""
Semantic Text Chunker
Splits Documents into overlapping Chunks while respecting sentence boundaries.

Strategy:
  1. Split document text into sentences.
  2. Greedily accumulate sentences until the chunk token budget is reached.
  3. On overflow, emit the current chunk and start a new one with the
     configured overlap (last N tokens of the previous chunk).
"""

from config import chunking_config
from utils.helpers import (
    generate_chunk_id,
    split_into_sentences,
    token_count_approx,
)
from utils.logger import logger
from utils.models import Chunk, Document


class SemanticChunker:
    """
    Split Documents into semantically coherent, token-bounded Chunks.

    Args:
        chunk_size:    Target number of tokens per chunk.
        chunk_overlap: Number of tokens to carry forward into the next chunk.
        min_chunk_size: Discard chunks below this token count.
    """

    def __init__(
        self,
        chunk_size: int = chunking_config.chunk_size,
        chunk_overlap: int = chunking_config.chunk_overlap,
        min_chunk_size: int = chunking_config.min_chunk_size,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ── Public API ────────────────────────────────────────────────────────────

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(
        self,
        documents: list[Document],
        tenant_id: str | None = None,
    ) -> list[Chunk]:
        """
        Chunk a list of Documents.

        Args:
            documents: Documents from the ingestion pipeline.
            tenant_id: Unique identifier for the tenant.

        Returns:
            Flat list of Chunk objects with metadata and unique IDs.
        """
        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        for doc in documents:
            chunks = self._chunk_document(doc, tenant_id=tenant_id)
            for c in chunks:
                c_id = c.chunk_id
                if c_id in seen_ids:
                    suffix = 1
                    while f"{c_id}_{suffix}" in seen_ids:
                        suffix += 1
                    c.chunk_id = f"{c_id}_{suffix}"
                seen_ids.add(c.chunk_id)
                all_chunks.append(c)

            logger.debug(
                f"Chunked '{doc.source}' page {doc.metadata.get('page', '?')} "
                f"→ {len(chunks)} chunk(s)"
            )
        logger.info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk_document(
        self,
        doc: Document,
        tenant_id: str | None = None,
    ) -> list[Chunk]:
        """Split a single Document into Chunks."""
        sentences = split_into_sentences(doc.content)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = token_count_approx(sentence)

            # If a single sentence exceeds the chunk size, hard-split it
            if sentence_tokens > self.chunk_size:
                if current_sentences:
                    chunks.append(
                        self._make_chunk(doc, current_sentences, chunk_index, tenant_id=tenant_id)
                    )
                    chunk_index += 1
                    current_sentences, current_tokens = self._carry_overlap(
                        current_sentences
                    )
                # Split the oversized sentence into word-level sub-chunks
                for sub in self._split_long_sentence(sentence):
                    current_sentences.append(sub)
                    current_tokens += token_count_approx(sub)
                    if current_tokens >= self.chunk_size:
                        chunks.append(
                            self._make_chunk(doc, current_sentences, chunk_index, tenant_id=tenant_id)
                        )
                        chunk_index += 1
                        current_sentences, current_tokens = self._carry_overlap(
                            current_sentences
                        )
                continue

            # Normal sentence fits — check if we need to flush
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunks.append(
                    self._make_chunk(doc, current_sentences, chunk_index, tenant_id=tenant_id)
                )
                chunk_index += 1
                current_sentences, current_tokens = self._carry_overlap(
                    current_sentences
                )

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Flush remainder
        if current_sentences:
            chunks.append(
                self._make_chunk(doc, current_sentences, chunk_index, tenant_id=tenant_id)
            )

        # Filter out tiny chunks
        return [c for c in chunks if token_count_approx(c.text) >= self.min_chunk_size]

    def _make_chunk(
        self,
        doc: Document,
        sentences: list[str],
        index: int,
        tenant_id: str | None = None,
    ) -> Chunk:
        """Assemble sentences into a Chunk with metadata."""
        import datetime
        from pathlib import Path

        text = " ".join(sentences).strip()
        eff_tenant = (
            tenant_id
            or doc.metadata.get("tenant_id")
            or "default_tenant"
        )
        filename = doc.metadata.get("filename") or Path(doc.source).name
        upload_ts = (
            doc.metadata.get("upload_timestamp")
            or datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        page = doc.metadata.get("page", 1)

        metadata = {
            **doc.metadata,
            "chunk_index": index,
            "token_count": token_count_approx(text),
            "tenant_id": eff_tenant,
            "filename": filename,
            "upload_timestamp": upload_ts,
        }

        return Chunk(
            text=text,
            source=doc.source,
            chunk_id=generate_chunk_id(doc.source, index, text, page=page),
            page=page,
            metadata=metadata,
        )



    def _carry_overlap(self, sentences: list[str]):
        """
        Return the suffix of sentences that sum to ~chunk_overlap tokens,
        along with their total token count.
        """
        overlap_sentences: list[str] = []
        token_sum = 0
        for sentence in reversed(sentences):
            t = token_count_approx(sentence)
            if token_sum + t > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            token_sum += t
        return overlap_sentences, token_sum

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Hard-split a very long sentence into word-level sub-chunks."""
        words = sentence.split()
        sub_chunks = []
        step = int(self.chunk_size * 0.75)  # ~words per sub-chunk
        for i in range(0, len(words), step):
            sub_chunks.append(" ".join(words[i : i + step]))
        return sub_chunks
