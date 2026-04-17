# 📚 Ask My Documents — Production RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Evaluation](https://img.shields.io/badge/eval-RAGAS-green.svg)](evaluation/)

A **production-grade Retrieval-Augmented Generation (RAG)** system that lets you upload documents and ask questions, with answers **strictly grounded in evidence** and **explicit citations**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📂 **Multi-format ingestion** | PDF, TXT, Markdown |
| ✂️ **Semantic chunking** | 500–800 tokens, ~100 token overlap |
| 🧠 **Sentence-Transformer embeddings** | `all-MiniLM-L6-v2`, disk-cached |
| 🗄️ **ChromaDB vector store** | Persistent, cosine similarity |
| 🔍 **Hybrid retrieval** | BM25 + vector search via RRF |
| 🎯 **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` |
| 💬 **Citation-enforced answers** | Claude / GPT-4 / DeepSeek / OpenRouter with hallucination guards |
| 📊 **Evaluation pipeline** | RAGAS faithfulness, correctness, relevance |
| 🖥️ **Streamlit UI** | Upload docs, chat with citations |
| 🖱️ **CLI** | Full command-line interface |

---

## 📁 Project Structure

```
rag/
├── config/
│   ├── __init__.py          # Config loader (typed accessors)
│   ├── settings.yaml        # System configuration
│   └── prompts.yaml         # Versioned prompts
├── ingestion/
│   ├── base_loader.py       # Abstract loader interface
│   ├── pdf_loader.py        # PyMuPDF + pypdf fallback
│   ├── text_loader.py       # TXT and Markdown loaders
│   └── ingestion_pipeline.py# Routing + directory ingestion
├── chunking/
│   └── chunker.py           # SemanticChunker (sentence-boundary aware)
├── embeddings/
│   └── embedding_engine.py  # Batch encoding + disk cache
├── retrieval/
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── bm25_retriever.py    # BM25Okapi keyword search
│   ├── reranker.py          # CrossEncoder reranker
│   └── hybrid_retriever.py  # RRF fusion + reranking orchestrator
├── generation/
│   └── answer_generator.py  # Anthropic / OpenAI answer generation
├── evaluation/
│   ├── evaluate.py          # Evaluation script (RAGAS + fallback)
│   └── golden_dataset.json  # Sample Q&A pairs
├── utils/
│   ├── models.py            # Shared dataclasses
│   ├── helpers.py           # Text utils, hashing, citations
│   └── logger.py            # Rotating file + console logger
├── examples/
│   └── quickstart.py        # End-to-end demo
├── docs/
│   └── sample_doc.txt       # Sample document for testing
├── pipeline.py              # RAGPipeline orchestrator
├── app.py                   # Streamlit web UI
├── cli.py                   # Command-line interface
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### 3. Run the example

```bash
python examples/quickstart.py
```

---

## 🖥️ Web UI (Streamlit)

```bash
streamlit run app.py
```

- Upload PDF / TXT / Markdown files via the sidebar
- Ask questions in the chat interface
- Citations appear below each answer
- Enable **Debug Mode** to inspect retrieved chunks and scores

---

## 🖱️ Command-Line Interface

```bash
# Ingest a single document
python cli.py ingest docs/report.pdf

# Ingest all documents in a directory
python cli.py ingest-dir docs/

# Ask a question
python cli.py query "What is the refund policy?"

# JSON output (for scripting)
python cli.py query "What is RAG?" --json

# Knowledge base stats
python cli.py stats

# Remove a document
python cli.py delete report.pdf
```

---

## ⚙️ Configuration

Edit `config/settings.yaml` to tune the pipeline:

```yaml
chunking:
  chunk_size: 600      # Target tokens per chunk
  chunk_overlap: 100   # Overlap between chunks

retrieval:
  top_k: 10            # Candidates per retriever
  top_n_rerank: 5      # Final chunks after reranking
  use_bm25: true       # Enable hybrid retrieval
  use_reranker: true   # Enable cross-encoder reranking

generation:
  provider: "openrouter"          # or "anthropic" | "openai" | "deepseek"
  model: "openrouter/free"        # Auto-routes free requests
  temperature: 0.1                # Low = more factual
```

### Prompt versioning

All prompts live in `config/prompts.yaml`. Modify `answer_prompt` to experiment with different instruction styles without touching Python code.

---

## 📊 Evaluation

### Run on the golden dataset

```bash
python evaluation/evaluate.py
```

### CI/CD mode (fails build if scores drop)

```bash
python evaluation/evaluate.py --fail-on-threshold
```

Thresholds are configured in `config/settings.yaml`:

```yaml
evaluation:
  faithfulness_threshold: 0.7
  answer_correctness_threshold: 0.6
  context_relevance_threshold: 0.5
  fail_build_on_threshold: true
```

### Extending the golden dataset

Add items to `evaluation/golden_dataset.json`:

```json
{
  "id": "q006",
  "question": "What is the warranty period?",
  "ground_truth": "The product carries a 12-month warranty…",
  "source_documents": ["manual.pdf"]
}
```

---

## 🧩 Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────┐
│           HybridRetriever                 │
│  ┌─────────────┐    ┌──────────────────┐  │
│  │ BM25 Search │    │  Vector Search   │  │
│  │  (keyword)  │    │  (ChromaDB)      │  │
│  └──────┬──────┘    └────────┬─────────┘  │
│         └────────┬───────────┘            │
│              RRF Fusion                   │
│                  │                        │
│         CrossEncoder Reranker             │
└───────────────────────────────────────────┘
                   │
                   ▼ Top-N chunks
         ┌─────────────────────┐
         │   AnswerGenerator   │
         │  (Claude / GPT-4)   │
         └─────────────────────┘
                   │
                   ▼
          Answer + Citations
```

---

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI API key (alternative) |
| `COHERE_API_KEY` | Cohere API key (optional reranker) |

---

## 🛡️ Hallucination Prevention

The system uses multiple layers to prevent hallucination:

1. **System prompt** — Strictly instructs the LLM to only use provided context
2. **Low temperature** — `0.1` for deterministic, factual responses
3. **Fallback response** — Returns "Insufficient information to answer" when context is empty or irrelevant
4. **Citation enforcement** — Every claim must be cited with `[N]` notation
5. **Evaluation gating** — CI/CD pipeline blocks deployment if faithfulness drops below threshold

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Sentence-Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [RAGAS](https://docs.ragas.io/)
- [rank_bm25](https://github.com/dorianbrown/rank_bm25)
- [Anthropic Claude](https://www.anthropic.com/)
