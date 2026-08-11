# 📚 Ask My Documents — Privacy-First RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React 18+](https://img.shields.io/badge/react-18+-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml/badge.svg?job=test)](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml)

A **privacy-hardened, production-grade Retrieval-Augmented Generation (RAG)** web application designed for private document intelligence. Features **strict zero data training compliance**, **local PII anonymization**, **Groq free inference**, **Ollama offline LLM support**, **HyDE**, **Multi-Query Expansion**, and **Self-RAG evaluation scoring**.

---

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🔒 Privacy & Zero-Training Guarantee](#-privacy--zero-training-guarantee)
- [⚖️ Why RAG vs. Pasting Docs into LLMs](#-why-rag-vs-pasting-documents-into-chatgptclaude)
- [🧩 Retrieval Architecture & ML Pipeline](#-retrieval-architecture--ml-pipeline)
- [🛡️ Intelligent Fallback & Citation Handling](#️-intelligent-fallback--citation-handling)
- [⚡ Quick Start & Running Locally](#-quick-start--running-locally)
- [🔐 Supported LLM Inference Engines](#-supported-llm-inference-engines)
- [⚙️ Environment Variables & Setup](#️-environment-variables--setup)
- [🧪 Testing & CI Compliance](#-testing--ci-compliance)

---

## ✨ Key Features

| Category | Capability & Technology Stack | Technical Details |
|---|---|---|
| 📂 **1. Ingestion & Docs** | Layout-Aware & Hyperlink Ingestion | Extracts per-page text & embedded hyperlinks via `PyMuPDF` (`fitz`), with `pypdf` fallback. |
| ✂️ **2. Chunking Strategy** | Semantic Pitch-Deck Chunker | Sentence-boundary aware regex splitting (`250` token size, `15` token min limit to preserve bullet points). |
| 🗄️ **3. Embeddings & Storage** | Dense Vectors & Multi-Tenant ChromaDB | `SentenceTransformers all-MiniLM-L6-v2` (384-dim) with isolated per-tenant vector collections (`tenant_<id>`). |
| 🔍 **4. Retrieval Engine** | Hybrid BM25 + Vector Search + Reranker | Combines `rank_bm25` and ChromaDB via RRF fusion + `ms-marco-MiniLM-L-6-v2` cross-encoder reranking. |
| 🛡️ **5. Privacy Layer** | Built-in Local PII Redaction & Zero-Training APIs | Scrubber sanitizes names, emails, IPs locally by default; routes requests exclusively to contractually zero-training APIs like Groq. |
| 🤖 **6. LLM Inference** | Groq, OpenAI, Anthropic, DeepSeek, Gemini, OpenRouter | Default: **Groq (Llama 3.3 70B)** for free, zero-training inference; supports Multi-Key Rotation and BYOK (Bring Your Own Key) for free & paid tiers. |
| 🖥️ **7. Modern Interfaces** | React + Vite UI, FastAPI Backend & Streamlit | Premium Parchment Editorial design UI (`frontend/`), FastAPI SSE streaming endpoints (`api/`), and legacy Streamlit app (`app.py`). |

---

## 🔒 Privacy & Zero-Training Guarantee

This application enforces a strict privacy-first architecture to ensure **your documents remain private and are NEVER used to train AI models**:

1. **Strict Provider Filtering**: Uses Groq API as the default cloud provider whose official terms guarantee zero data retention and zero training on API payloads.
2. **Built-in Local PII Anonymization**: Client-side regex scrubber (`utils/anonymizer.py`) automatically sanitizes personal names, email addresses, phone numbers, and IP addresses *before* payload transmission to any external provider.
3. **Multi-Key Rotation & BYOK**: Supports comma-separated API keys (`GROQ_API_KEY=key1,key2,key3`) for rate limit rotation, as well as Bring Your Own Key (BYOK) for any free or paid API key across Groq, OpenAI, Anthropic Claude, DeepSeek, Google Gemini, and OpenRouter.
4. **No Account / Cookie-Based Multi-Tenancy**: Anonymous `tenant_id` stored in `HttpOnly` browser cookies isolates database vector collections per user session without requesting email or login credentials.
5. **Data Control**: Delete individual documents or execute a full tenant data wipe with one click.

---

## ⚖️ Why RAG vs. Pasting Documents into ChatGPT/Claude

| Dimension | Pasting Docs into Consumer LLMs | Production Privacy-Hardened RAG |
|---|---|---|
| **Privacy Policy** | Consumer free tiers may retain and train on pasted conversation data. | Guaranteed zero training data retention + local PII redaction. |
| **Corpus Capacity** | Restricted by prompt context window; large document sets get truncated. | Unlimited document corpus scaled across persistent vector database. |
| **Source Citations** | Vague or non-existent; cannot verify exact page source. | Enforced `[N]` citations with page numbers, excerpts, and split-screen inspector. |
| **Hallucination Guard** | LLMs guess or hallucinate when facts are missing from prompt context. | Strict ground-truth prompt guards + automatic fallback response when context is missing. |
| **Retrieval Quality** | Context dump causes "lost-in-the-middle" attention degradation. | BM25 + Dense Vector + Cross-Encoder reranking feeds only high-relevance evidence chunks. |

---

## 🧩 Retrieval Architecture & ML Pipeline

1. **Hybrid Retrieval**:
   - **BM25 Keyword Search**: Captures exact terminology, acronyms, product names, and numerical values.
   - **Dense Vector Search**: Captures semantic intent using `all-MiniLM-L6-v2` embeddings.
   - **Reciprocal Rank Fusion (RRF)**: Fuses keyword and vector rankings into a unified score.
   - **Cross-Encoder Reranking**: Re-scores top candidates using `ms-marco-MiniLM-L-6-v2` for precise filtering.

2. **HyDE (Hypothetical Document Embeddings)**:
   - LLM generates a hypothetical target answer, which is embedded to bridge the gap between user questions and formal document text.

3. **Multi-Query Expansion**:
   - Reformulates questions into semantic variations to maximize retrieval recall.

---

## 🛡️ Intelligent Fallback & Citation Handling

- **Ungrounded Questions**: When a query cannot be answered from your uploaded documents, the system returns a friendly, standardized response:
  > *"I could not find relevant information in your uploaded documents to answer this question. Please upload a document containing details on this topic or rephrase your query."*
- **Citation Suppression**: Fallback answers automatically suppress document citations (`citations: []`), avoiding misleading footers when information is missing.

---

## ⚡ Quick Start & Running Locally

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Sudhanshub27/rag-system.git
cd rag-system

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and insert your free Groq API key:

```bash
cp .env.example .env
```

Example `.env`:
```env
GROQ_API_KEY=gsk_your_free_groq_key_here
```

### 3. Launch Development Servers

**Backend API (FastAPI)**:
```bash
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend App (React + Vite)**:
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 🔐 Supported LLM Inference Engines
- **Groq API** — *Default Free Tier (Zero Training Guaranteed)*
- **OpenAI API** — *Supports All Free & Paid GPT Models*
- **Anthropic Claude API** — *Supports All Free & Paid Claude Models*
- **DeepSeek API** — *Supports All DeepSeek Models*
- **Google Gemini API** — *Supports All Gemini Models*
- **OpenRouter API** — *Multi-Model Hub for All Open-Source & Closed Models*

---

## 🧪 Testing & CI Compliance

Run the automated test suite (35 unit & integration tests):

```bash
# Run pytest test suite
.venv/bin/pytest tests/

# Run linter checks
.venv/bin/ruff check .
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
