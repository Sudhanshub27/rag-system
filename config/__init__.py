"""
RAG System – Core Configuration Loader
Loads settings.yaml and prompts.yaml and provides typed access throughout
the application. All modules import from here rather than reading YAML directly.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load .env first so env vars are available
load_dotenv()

# Resolve the config directory relative to this file
_CONFIG_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load and parse a YAML config file."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Load configs once at import time ──────────────────────────────────────────
_settings: Dict[str, Any] = _load_yaml("settings.yaml")
_prompts: Dict[str, Any] = _load_yaml("prompts.yaml")


# ── Typed accessors ──────────────────────────────────────────────────────────

class EmbeddingConfig:
    model_name: str = _settings["embeddings"]["model_name"]
    device: str = _settings["embeddings"]["device"]
    batch_size: int = _settings["embeddings"]["batch_size"]
    cache_dir: str = _settings["embeddings"]["cache_dir"]


class ChunkingConfig:
    chunk_size: int = _settings["chunking"]["chunk_size"]
    chunk_overlap: int = _settings["chunking"]["chunk_overlap"]
    min_chunk_size: int = _settings["chunking"]["min_chunk_size"]


class VectorStoreConfig:
    provider: str = _settings["vector_store"]["provider"]
    persist_directory: str = _settings["vector_store"]["persist_directory"]
    collection_name: str = _settings["vector_store"]["collection_name"]


class RetrievalConfig:
    top_k: int = _settings["retrieval"]["top_k"]
    top_n_rerank: int = _settings["retrieval"]["top_n_rerank"]
    use_bm25: bool = _settings["retrieval"]["use_bm25"]
    bm25_weight: float = _settings["retrieval"]["bm25_weight"]
    vector_weight: float = _settings["retrieval"]["vector_weight"]
    reranker_model: str = _settings["retrieval"]["reranker_model"]
    use_reranker: bool = _settings["retrieval"]["use_reranker"]


class GenerationConfig:
    provider: str = _settings["generation"]["provider"]
    model: str = _settings["generation"]["model"]
    max_tokens: int = _settings["generation"]["max_tokens"]
    temperature: float = _settings["generation"]["temperature"]
    timeout_seconds: int = _settings["generation"]["timeout_seconds"]


class LoggingConfig:
    level: str = _settings["logging"]["level"]
    log_file: str = _settings["logging"]["log_file"]
    max_file_size_mb: int = _settings["logging"]["max_file_size_mb"]
    backup_count: int = _settings["logging"]["backup_count"]


class IngestionConfig:
    supported_extensions: list = _settings["ingestion"]["supported_extensions"]
    max_file_size_mb: int = _settings["ingestion"]["max_file_size_mb"]


class EvaluationConfig:
    golden_dataset_path: str = _settings["evaluation"]["golden_dataset_path"]
    output_path: str = _settings["evaluation"]["output_path"]
    faithfulness_threshold: float = _settings["evaluation"]["faithfulness_threshold"]
    answer_correctness_threshold: float = _settings["evaluation"]["answer_correctness_threshold"]
    context_relevance_threshold: float = _settings["evaluation"]["context_relevance_threshold"]
    fail_build_on_threshold: bool = _settings["evaluation"]["fail_build_on_threshold"]


class PromptsConfig:
    system_prompt: str = _prompts["system_prompt"]
    retrieval_prompt: str = _prompts["retrieval_prompt"]
    answer_prompt: str = _prompts["answer_prompt"]
    rerank_instruction: str = _prompts["rerank_instruction"]
    fallback_response: str = _prompts["fallback_response"]
    thresholds: Dict[str, Any] = _prompts["thresholds"]
    diagram_system_prompt: str = _prompts.get("diagram_system_prompt", "You are a Mermaid diagram expert. Output ONLY valid Mermaid syntax.")
    diagram_prompt: str = _prompts.get("diagram_prompt", "Generate a Mermaid {diagram_type} diagram from the context below.\n\nContext:\n{context}\n\nRequest: {question}\n\nMermaid Diagram:")


# ── API Keys (from environment) ───────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# ── Singleton instances ───────────────────────────────────────────────────────
embedding_config = EmbeddingConfig()
chunking_config = ChunkingConfig()
vector_store_config = VectorStoreConfig()
retrieval_config = RetrievalConfig()
generation_config = GenerationConfig()
logging_config = LoggingConfig()
ingestion_config = IngestionConfig()
evaluation_config = EvaluationConfig()
prompts_config = PromptsConfig()
