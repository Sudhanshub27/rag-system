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
- [🧭 Query Routing (Narrow vs Broad)](#-query-routing-narrow-vs-broad)
- [🔐 Data Handling & Security](#-data-handling--security)
- [🛡️ Intelligent Fallback & Citation Handling](#️-intelligent-fallback--citation-handling)
- [⚡ Quick Start & Running Locally](#-quick-start--running-locally)
- [🔐 Supported LLM Inference Engines](#-supported-llm-inference-engines)
- [🧪 Testing & CI Compliance](#-testing--ci-compliance)

---


## ✨ Key Features

| Category | Capability & Technology Stack | Technical Details |
|---|---|---|
| 📂 **1. Ingestion & Docs** | Layout-Aware & Hyperlink Ingestion | Extracts per-page text & embedded hyperlinks via `PyMuPDF` (`fitz`), with `pypdf` fallback. |
| ✂️ **2. Chunking Strategy** | Semantic Pitch-Deck Chunker | Sentence-boundary aware regex splitting (`250` token size, `15` token min limit to preserve bullet points). |
| 🗄️ **3. Embeddings & Storage** | Dense Vectors & Multi-Tenant ChromaDB | `ONNX Runtime / SentenceTransformers all-MiniLM-L6-v2` (384-dim) with isolated per-tenant vector collections (`tenant_<id>`). |
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
   - **Dense Vector Search**: Captures semantic intent using `BAAI/bge-base-en-v1.5` embeddings.
   - **Reciprocal Rank Fusion (RRF)**: Fuses keyword and vector rankings into a unified score.
   - **Cross-Encoder Reranking**: Re-scores top candidates using `ms-marco-MiniLM-L-6-v2` for precise filtering.

2. **HyDE (Hypothetical Document Embeddings)**:
   - LLM generates a hypothetical target answer, which is embedded to bridge the gap between user questions and formal document text.

3. **Multi-Query Expansion**:
   - Reformulates questions into semantic variations to maximize retrieval recall.

---

## 🧭 Query Routing (Narrow vs Broad)

Specific lookup questions ("what is the refund policy?") and whole-document
questions ("explain this document") need different pipelines. Retrieval-based
top-k chunking works well for the former and breaks down for the latter, since
no fixed set of chunks can represent an 80-page document.

`retrieval/query_router.py` classifies each query before retrieval runs:

1. **Pattern match** (free) — keywords like *explain / summarize / overview /
   walk me through* flag a query as broad.
2. **BM25 score-shape fallback** (free) — if scores are flat across many
   chunks rather than peaked on a few, the query is broad.
3. **Model classification** (last resort, cheap call) — only if 1 and 2 are
   inconclusive.

| Query type | Path | Cost |
|---|---|---|
| Narrow | `HybridRetriever` → top-N chunks → 1 generation call | 1 call, every time |
| Broad (cached) | Cached doc-level summary → 1 generation call | 1 call |
| Broad (first time) | Map-reduce over all chunks → cache → generation call | ~15–20 calls, once per doc |

### Broad-query handling: map-reduce summarization

`generation/doc_summarizer.py` builds a document-level summary once, on the
first broad query for a given document:

- **Map**: chunks are grouped (~4–5 chunks/group) and each group is
  summarized in a separate, rate-limited call.
- **Reduce**: group summaries are combined into a final document summary and
  section outline.
- **Cache**: the result is stored keyed by a content hash of the source
  document (not filename), so any edit to the source automatically
  invalidates the cached summary.
- Every subsequent "explain this document" query is served straight from
  cache — no further generation calls until the source document changes.

API calls are rate-limited (semaphore + backoff honoring `retry-after`) to
stay within the configured provider's RPM/TPM limits.

---

## 🔐 Data Handling & Security

This project processes documents via third-party inference APIs
(Groq / Anthropic / OpenAI, depending on configuration) and, in this
deployment, is hosted on Oracle Cloud Infrastructure. Data necessarily
passes through that infrastructure to be processed — no cloud-based
system can generate an answer without the model reading the input.
What follows is what actually protects that data, rather than an
absolute "never leaves the system" claim, which would not be accurate
for any cloud-connected app.

- **In transit**: all API calls (to the LLM provider, to Oracle-hosted
  endpoints) use TLS/HTTPS only.
- **At rest**: documents, chunks, embeddings, and cached summaries stored
  on Oracle Cloud use encryption at rest (Oracle Object/Block Storage
  default encryption).
- **Secrets**: API keys are never committed; local development uses
  `.env` (see `.env.example`), production deployments should use a
  secrets manager (e.g. Oracle Vault) rather than plaintext env files.
- **Cache keys**: cached summaries are keyed by a content hash of the
  document, not filenames or raw identifiers.
- **Access control**: Oracle-hosted storage/DB is restricted to the
  application's network security group; no public inbound access.
- **Inference provider data use**: per Groq's Services Agreement, inputs
  and outputs are not used for training/fine-tuning without explicit
  permission, and are not retained beyond what's needed to serve the
  request, transient reliability/abuse-monitoring logs (up to 30 days),
  or legal requirements. Zero Data Retention can be enabled in the
  provider's console for stricter handling. Equivalent terms apply if
  configured to use Anthropic or OpenAI instead — check the active
  provider's current DPA before deploying with sensitive documents.

This is standard "controlled, encrypted, access-restricted, contractually
bound" data handling — the same model every major cloud and inference
provider operates under. No provider, including this one, can offer an
absolute guarantee that data is physically inaccessible to their own
infrastructure; the guarantee that matters is that access is encrypted,
logged, audited, and contractually restricted from being used or shared
beyond serving the request.



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
