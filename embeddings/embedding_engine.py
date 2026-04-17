"""
Embedding Engine
Wraps Sentence-Transformers to produce consistent embeddings for chunks and queries.

Features:
  - Batch encoding with configurable batch size
  - Disk-level embedding cache (JSON-based) to avoid redundant computation
  - Thread-safe model loading (loaded once at construction)
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import embedding_config
from utils.logger import logger
from utils.models import Chunk


class EmbeddingEngine:
    """
    Embed text using a Sentence-Transformers model.

    Args:
        model_name:  HuggingFace model identifier.
        device:      'cpu' or 'cuda'.
        batch_size:  Number of texts per encoding batch.
        cache_dir:   Directory to cache computed embeddings.
    """

    def __init__(
        self,
        model_name: str = embedding_config.model_name,
        device: str = embedding_config.device,
        batch_size: int = embedding_config.batch_size,
        cache_dir: Optional[str] = embedding_config.cache_dir,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = self.cache_dir / f"{self._safe_model_name()}_cache.json"
            self._cache: dict = self._load_cache()
        else:
            self._cache = {}

        logger.info(f"Loading embedding model: {model_name} on {device}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Embed a list of Chunks in batches.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of embedding vectors (one per chunk).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of raw strings, using the disk cache when possible.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                embeddings[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        logger.debug(
            f"Embedding {len(texts)} texts — "
            f"{len(uncached_texts)} cache misses, "
            f"{len(texts) - len(uncached_texts)} cache hits"
        )

        if uncached_texts:
            new_vectors = self._encode_batch(uncached_texts)
            for idx, vector in zip(uncached_indices, new_vectors):
                embeddings[idx] = vector
                self._cache[self._hash(texts[idx])] = vector

            self._save_cache()

        return embeddings  # type: ignore[return-value]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: User question text.

        Returns:
            Embedding vector.
        """
        return self.embed_texts([query])[0]

    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._model.get_sentence_embedding_dimension()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Run the Sentence-Transformer model in batches."""
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2-normalize for cosine sim
            )
            return [v.tolist() for v in vectors]
        except Exception as e:
            logger.error(f"Embedding batch failed: {e}")
            raise RuntimeError(f"Embedding failed: {e}") from e

    @staticmethod
    def _hash(text: str) -> str:
        """SHA-256 hash of text for cache keying (first 32 chars)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _safe_model_name(self) -> str:
        """Convert model name to a filesystem-safe string."""
        return self.model_name.replace("/", "_").replace("\\", "_")

    def _load_cache(self) -> dict:
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r") as f:
                    data = json.load(f)
                logger.debug(f"Loaded {len(data)} cached embeddings")
                return data
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
        return {}

    def _save_cache(self) -> None:
        if self.cache_dir:
            try:
                with open(self._cache_file, "w") as f:
                    json.dump(self._cache, f)
            except Exception as e:
                logger.warning(f"Could not save embedding cache: {e}")
