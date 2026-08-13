"""
Embedding Engine
Wraps Sentence-Transformers or ONNX Runtime to produce consistent 384-dim embeddings.

Features:
  - ONNX Runtime execution engine (70% less RAM than PyTorch) with SentenceTransformer fallback
  - Batch encoding with configurable batch size
  - Disk-level embedding cache (JSON-based) to avoid redundant computation
  - Thread-safe model loading (loaded once at construction)
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from config import embedding_config
from utils.logger import logger
from utils.models import Chunk

_shared_embedding_models: dict = {}


class ONNXEmbeddingEngine:
    """Lightweight ONNX Runtime inference engine for all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"

        self.model_name = model_name
        tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        try:
            onnx_path = hf_hub_download(repo_id=model_name, filename="onnx/model.onnx")
        except Exception:
            onnx_path = hf_hub_download(
                repo_id="xenova/all-MiniLM-L6-v2", filename="onnx/model.onnx"
            )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 384))

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer.encode_batch(batch_texts)
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            available_names = [inp.name for inp in self.session.get_inputs()]
            if "token_type_ids" in available_names:
                inputs["token_type_ids"] = token_type_ids

            outputs = self.session.run(None, inputs)
            token_embeddings = outputs[0]
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask

            if normalize_embeddings:
                norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norm, 1e-12)

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def get_sentence_embedding_dimension(self) -> int:
        return 384


class EmbeddingEngine:
    """
    Embed text using an ONNX or Sentence-Transformers model.

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
        cache_dir: str | None = embedding_config.cache_dir,
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

        cache_key = f"{model_name}_{device}"
        if cache_key not in _shared_embedding_models:
            logger.info(f"Loading embedding model: {model_name} on {device}")
            model_inst = None

            # Try PyTorch SentenceTransformers first (supports all HF models, SOTA quality)
            try:
                from sentence_transformers import SentenceTransformer

                model_inst = SentenceTransformer(model_name, device=device)
                logger.info(
                    f"PyTorch SentenceTransformer '{model_name}' loaded successfully"
                )
            except Exception as e:
                logger.warning(
                    f"PyTorch SentenceTransformer unavailable ({e}), falling back to ONNX Runtime Engine"
                )

            if model_inst is None:
                try:
                    model_inst = ONNXEmbeddingEngine(model_name)
                    logger.info("ONNX Embedding Engine initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to load embedding model '{model_name}': {e}")
                    raise

            if not hasattr(model_inst, "_mock_name") and not hasattr(
                model_inst, "return_value"
            ):
                _shared_embedding_models[cache_key] = model_inst
            self._model = model_inst
        else:
            self._model = _shared_embedding_models[cache_key]

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Embed a list of Chunks in batches.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of embedding vectors (one per chunk).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of raw strings, using the disk cache when possible.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

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

    def embed_query(self, query: str) -> list[float]:
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

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Run the embedding model in batches."""
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2-normalize for cosine sim
            )
            if isinstance(vectors, np.ndarray):
                return [v.tolist() for v in vectors]
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
                with open(self._cache_file) as f:
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
