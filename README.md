# 📚 Ask My Documents — Privacy-First RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React 18+](https://img.shields.io/badge/react-18+-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: Source-Available](https://img.shields.io/badge/License-Source_Available-red.svg)](LICENSE)
[![Privacy Policy](https://img.shields.io/badge/Privacy_Policy-Enforced-green.svg)](PRIVACY_POLICY.md)
[![Tests](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml/badge.svg?job=test)](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml)

A **privacy-hardened, production-grade Retrieval-Augmented Generation (RAG)** web application built with a **FastAPI backend** and **React + Vite frontend**. Designed for private document intelligence, it features **strict multi-tenant data isolation**, **zero data-training compliance**, **local client-side PII anonymization**, **Groq free inference**, **HyDE**, **Multi-Query Expansion**, and **Self-RAG evaluation scoring**.

---

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🔒 Privacy & Zero-Training Guarantee](#-privacy--zero-training-guarantee)
- [🏢 Multi-Tenant Isolation Architecture](#-multi-tenant-isolation-architecture)
- [⚖️ Why RAG vs. Pasting Docs into LLMs](#-why-rag-vs-pasting-documents-into-chatgptclaude)
- [🧩 Retrieval Architecture & ML Pipeline](#-retrieval-architecture--ml-pipeline)
- [🧭 Query Routing (Narrow vs Broad)](#-query-routing-narrow-vs-broad)
- [🌐 SEO Metadata & Search Engine Crawling](#-seo-metadata--search-engine-crawling)
- [🔐 Data Handling & Security](#-data-handling--security)
- [📂 Project Structure](#-project-structure)
- [⚡ Quick Start & Running Locally](#-quick-start--running-locally)
- [🐳 Docker & Cloud Deployment](#-docker--cloud-deployment)
- [🔐 Supported LLM Inference Engines](#-supported-llm-inference-engines)
- [🧪 Testing & CI Compliance](#-testing--ci-compliance)
- [📜 Privacy Policy & Terms of Use](#-privacy-policy--terms-of-use)
- [📝 License & Code Usage Terms](#-license--code-usage-terms)

---

## ✨ Key Features

| Category | Capability & Technology Stack | Technical Details |
|---|---|---|
| 🖥️ **1. Modern Web Application** | React 18 + Vite & FastAPI SSE Streaming | Parchment Editorial UI (`frontend/`) with real-time SSE progress streaming (`api/routes/upload.py`) & SSE token response generation (`api/routes/query.py`). |
| 🏢 **2. Multi-Tenant Isolation** | Physical Collection & Cache Isolation | Anonymous `HttpOnly` cookie-based `tenant_id` scopes ChromaDB collections (`tenant_<id>`), BM25 keyword indices, and summary caches. |
| 📂 **3. Ingestion & Document Processing** | Layout-Aware & Hyperlink Extraction | Page-by-page text & link parsing via `PyMuPDF` (`fitz`), fallback to `pypdf`, handling PDF, TXT, and Markdown files. |
| ✂️ **4. Chunking Strategy** | Semantic Pitch-Deck Chunker | Sentence-boundary aware regex splitting (`250` token target, `15` token min limit to preserve bullet points and lists). |
| 🗄️ **5. Embeddings & Storage** | Dense Vectors & Multi-Tenant ChromaDB | `SentenceTransformers all-MiniLM-L6-v2` (384-dim ONNX optimized embeddings) stored in tenant-scoped vector containers. |
| 🔍 **6. Hybrid Retrieval Engine** | BM25 + Vector Search + Cross-Encoder | Reciprocal Rank Fusion (RRF) combining `rank_bm25` and ChromaDB vector search, reranked via `ms-marco-MiniLM-L-6-v2`. |
| 🧭 **7. Dual Query Router** | Intent-Driven Processing | Distinguishes specific fact lookups (Narrow) from whole-document overview requests (Broad) using Map-Reduce summarization. |
| 🛡️ **8. Privacy & PII Scrubbing** | Client-Side Regex Anonymization | Local scrubber sanitizes personal names, email addresses, phone numbers, and IP addresses prior to API payload transmission. |
| 🤖 **9. LLM Inference Engine** | Groq, OpenAI, Anthropic, DeepSeek, Gemini, OpenRouter | Default: **Groq (Llama 3.3 70B)** for free, zero-training inference; supports Multi-Key Rotation and BYOK (Bring Your Own Key) for free & paid tiers. |

---

## 🔒 Privacy & Zero-Training Guarantee

This application enforces a strict privacy-first architecture to ensure **your documents remain private and are NEVER used to train AI models**:

1. **Strict Provider Filtering**: Uses Groq API as the default cloud provider whose official terms guarantee zero data retention and zero training on API payloads.
2. **Built-in Local PII Anonymization**: Client-side regex scrubber (`utils/anonymizer.py`) automatically sanitizes personal names, email addresses, phone numbers, and IP addresses *before* payload transmission to any external provider.
3. **Multi-Key Rotation & BYOK**: Supports comma-separated API keys (`GROQ_API_KEY=key1,key2,key3`) for rate limit rotation, as well as Bring Your Own Key (BYOK) for any free or paid API key across Groq, OpenAI, Anthropic Claude, DeepSeek, Google Gemini, and OpenRouter.
4. **No Account / Cookie-Based Multi-Tenancy**: Anonymous `tenant_id` stored in `HttpOnly` browser cookies isolates database vector collections per user session without requesting email or login credentials.
5. **Data Control**: Delete individual documents or execute a full tenant data wipe with one click.
6. **Code Usage Notice**: Source code is open on GitHub for inspection, but is **not open source**. Code cannot be copied, modified, distributed, or deployed without explicit written consent. See [PRIVACY_POLICY.md](PRIVACY_POLICY.md).

---

## 🏢 Multi-Tenant Isolation Architecture

The system achieves structural data isolation across multiple concurrent tenants without requiring account creation:

```
                  ┌─────────────────────────────────────────┐
                  │          HTTP Request Header            │
                  │   Cookie: rag_tenant_id = "tenant_123"  │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend (api/)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Vector Store:    ChromaDB Collection -> tenant_tenant_123                │
│  • Keyword Index:   BM25 Retriever      -> In-Memory tenant_123 Scope        │
│  • Cache Store:     DocSummarizer       -> summary_tenant_123_{hash}.json   │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Per-Tenant Vector Collections**: Every ChromaDB collection is physically isolated under `tenant_{tenant_id}`. A query from Tenant A cannot access document embeddings of Tenant B.
2. **Per-Tenant BM25 Keyword Indices**: Sparse BM25 keyword search indices are scoped per tenant context.
3. **Per-Tenant Summary Caches**: Cached document summaries generated during Map-Reduce processing are saved to `.cache/summaries/summary_{tenant_id}_{hash}.json`.
4. **Instant Multi-Tenant Purging**: Deleting data via `/api/tenant/{tenant_id}` drops the isolated ChromaDB collection, clears the tenant's BM25 index, and removes associated summary caches from disk.

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

```
  Document Upload ──► PyMuPDF Ingestion ──► Semantic Chunker (250 tokens)
                                                    │
                                                    ▼
  User Query ◄── Cross-Encoder Reranker ◄── RRF Fusion ◄── Dense Vector (MiniLM)
       │                                                      +
       ▼                                                 Sparse BM25 Keyword
  Answer Generator ──► SSE Token Stream ──► React Reading Pane + Inspector
```

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

## 🧭 Query Routing (Narrow vs Broad)

Specific lookup questions ("what is the refund policy?") and whole-document questions ("explain this document") use different execution pipelines.

`retrieval/query_router.py` classifies each query before retrieval runs:

1. **Pattern match** (zero cost) — Regex patterns for *explain, summarize, overview, break down, most important section, deep dive* flag a query as `BROAD`.
2. **BM25 score-shape fallback** (zero cost) — If scores are flat across many chunks rather than peaked on a few, the query is treated as `BROAD`.
3. **Model classification** (last resort) — Lightweight classification if pattern and score heuristics are ambiguous.

| Query Type | Execution Path | LLM Cost |
|---|---|---|
| **Narrow** | `HybridRetriever` → Top-N chunks → Reranker → LLM Answer | 1 generation call |
| **Broad (Cached)** | `.cache/summaries/summary_{tenant_id}_{hash}.json` | 0ms, 1 generation call |
| **Broad (First time)** | Parallel Map-Reduce (`ThreadPoolExecutor`) over 12-chunk groups | ~2–4 calls, cached forever |

---

## 🌐 SEO Metadata & Search Engine Crawling

The application includes full web metadata and search engine crawling configuration for optimal discovery and accessibility:

- **Sitemap (`sitemap.xml`)**: Configured in `frontend/public/sitemap.xml` and root `sitemap.xml` to provide search crawlers with structured URL priority mapping and update frequencies.
- **Robots Directives (`robots.txt`)**: Configured in `frontend/public/robots.txt` and root `robots.txt` to permit search engine indexers while restricting crawler access to private API endpoints (`/api/`) and temporary upload caches (`/tmp_uploads/`).
- **Rich Meta Tags**: `frontend/index.html` includes Open Graph (`og:*`), Twitter Card metadata, author credentials, canonical link tags, and structured sitemap declarations.

---

## 🔐 Data Handling & Security

This project processes documents via third-party inference APIs (Groq default; OpenAI, Anthropic, Gemini optional) hosted on containerized infrastructure (Docker / Cloud VM).

- **In transit**: All API calls use TLS/HTTPS encryption.
- **At rest**: Vector databases, embeddings, and caches use disk encryption at rest.
- **Secrets Management**: API keys are passed via environment variables (`.env` locally, Docker environment in production); keys are never logged or committed.
- **Access Control**: Multi-tenant session cookies ensure isolated data spaces for concurrent browser sessions.
- **Inference Provider Terms**: Per Groq's Services Agreement, API inputs/outputs are contractually restricted from being used for AI training or model fine-tuning.

---

## 📂 Project Structure

```
rag-system/
├── api/                        # FastAPI Web Backend
│   ├── main.py                 # FastAPI application entry point
│   ├── deps.py                 # Dependency injection & multi-tenant session management
│   └── routes/                 # SSE API endpoints (upload, query, documents, stats)
├── frontend/                   # React 18 + Vite Frontend App
│   ├── public/                 # Static web assets
│   │   ├── robots.txt          # Crawler instructions & API restrictions
│   │   └── sitemap.xml         # Site map XML for web metadata
│   ├── src/                    # Components (Navbar, Sidebar, ReadingPane, SourceInspector)
│   ├── index.html              # Main HTML page with OpenGraph & SEO tags
│   ├── vite.config.js          # Vite configuration & API proxy
│   └── package.json            # Frontend dependencies
├── ingestion/                  # Document Ingestion Pipeline
│   ├── pdf_loader.py           # PyMuPDF text & hyperlink parser
│   └── text_loader.py          # TXT & Markdown parser
├── chunking/                   # Text Segmentation
│   └── chunker.py              # Semantic regex pitch-deck chunker
├── embeddings/                 # Vector Embeddings
│   └── embedding_engine.py     # SentenceTransformers all-MiniLM-L6-v2 ONNX engine
├── retrieval/                  # Retrieval Engine
│   ├── vector_store.py         # Multi-tenant ChromaDB store wrapper
│   ├── bm25_retriever.py       # Scoped BM25 keyword indexer
│   ├── hybrid_retriever.py     # RRF Fusion & retrieval orchestrator
│   ├── reranker.py             # Cross-Encoder candidate reranker
│   └── query_router.py         # Intent classifier (Narrow vs Broad)
├── generation/                 # Response Generation
│   ├── answer_generator.py     # LLM answer generator & SSE stream provider
│   ├── doc_summarizer.py       # Parallel Map-Reduce cached document summarizer
│   └── diagram_generator.py    # Mermaid.js diagram generator
├── evaluation/                 # Evaluation & Benchmarks
│   ├── evaluate.py             # RAG triad evaluation runner
│   └── golden_dataset.json     # Ground truth evaluation dataset
├── utils/                      # Helper Utilities
│   ├── anonymizer.py           # Local client-side PII regex scrubber
│   ├── logger.py               # Structured logging system
│   └── rate_limiter.py         # Provider rate limiter & backoff handler
├── tests/                      # PyTest Test Suite
│   ├── unit/                   # Unit tests (isolation, router, chunker, generator)
│   └── integration/            # End-to-end integration tests
├── scripts/                    # Automation Scripts
│   └── ci_check.sh             # Local pre-push CI test script
├── LICENSE                     # Proprietary Source-Available License
├── PRIVACY_POLICY.md           # Privacy commitments & code usage terms
├── robots.txt                  # Root web crawler instructions
├── sitemap.xml                 # Root sitemap XML metadata
├── pipeline.py                 # Main RAGPipeline facade
├── cli.py                      # Multi-tenant CLI tool
├── docker-compose.yml          # Multi-container Docker orchestrator
├── Dockerfile.backend          # Backend FastAPI Dockerfile
├── Dockerfile.frontend         # Frontend React Vite Dockerfile
└── run.sh                      # Unified application control runner
```

---

## ⚡ Quick Start & Running Locally

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Sudhanshub27/rag-system.git
cd rag-system

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your free Groq API key:

```bash
cp .env.example .env
```

Example `.env`:
```env
GROQ_API_KEY=gsk_your_free_groq_key_here
```

### 3. Launch Application

#### Option A: Convenient Unified Script
```bash
./run.sh
```

#### Option B: Manual Local Development

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

### 4. Multi-Tenant CLI Usage

You can also run ingestion and query operations directly from the command line:

```bash
# Ingest document under tenant_org_a
python cli.py --tenant-id tenant_org_a ingest docs/sample_report.pdf

# Query document index for tenant_org_a
python cli.py --tenant-id tenant_org_a query "What are the Q3 financial results?"
```

---

## 🐳 Docker & Cloud Deployment

### 1. Docker Compose (Local or Server)

Run both the FastAPI backend and React frontend in containerized isolation:

```bash
# Build and start all services
docker-compose up --build -d

# Stop services
docker-compose down
```

Services will be available at:
- **Frontend App**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000/api`

---

### 2. Oracle Cloud Always Free Tier Deployment

The application is light-weight and optimized to run inside Oracle Cloud's Always Free VM instances (`Ampere A1` 4 ARM vCPUs / 24GB RAM or `E2.1.Micro` 1GB RAM instance):

1. **Provision VM**: Launch an Ubuntu VM on Oracle Cloud Infrastructure.
2. **Install Docker & Docker Compose**:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose
   ```
3. **Clone & Configure**:
   ```bash
   git clone https://github.com/Sudhanshub27/rag-system.git
   cd rag-system
   cp .env.example .env
   # Edit .env to set your GROQ_API_KEY
   ```
4. **Deploy Containers**:
   ```bash
   docker-compose up -d --build
   ```

---

## 🔐 Supported LLM Inference Engines

The application features a flexible LLM provider architecture with zero-training compliance:

- **Groq API** — *Default Free & Paid Models (Llama 3.3 70B, DeepSeek R1)*
- **OpenAI API** — *Supports GPT-4o, GPT-4o-mini, GPT-3.5-Turbo*
- **Anthropic Claude API** — *Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku*
- **DeepSeek API** — *Supports DeepSeek-V3, DeepSeek-R1*
- **Google Gemini API** — *Supports Gemini 1.5 Pro, Gemini 1.5 Flash*
- **OpenRouter API** — *Multi-Model Aggregator for Open & Commercial Models*

---

## 🧪 Testing & CI Compliance

Run the automated suite (unit tests, coverage, and linting):

```bash
# Run automated pre-push checks (Linter + Formatter + PyTest)
./scripts/ci_check.sh

# Or run pytest manually
.venv/bin/pytest tests/
```

---

## 📜 Privacy Policy & Terms of Use

Please read [PRIVACY_POLICY.md](PRIVACY_POLICY.md) for full details on:
- **Zero-Training AI Commitments**: Guaranteed data protection on API inputs/outputs.
- **Client-Side PII Scrubbing**: Redaction of personal info prior to transmission.
- **Multi-Tenant Cookie Isolation**: Scoped database collections per user session.
- **Code Usage Restrictions**: Source code is open for inspection, but **NOT open source** and cannot be used, modified, or deployed without consent.

---

## 📝 License & Code Usage Terms

Copyright (c) 2026 Sudhanshu Batra. All Rights Reserved.

This project is licensed under a **Source-Available / Proprietary License** — see [LICENSE](LICENSE) for details.

- **Open Code Access**: The source code is publicly accessible on GitHub for transparency, security inspection, and educational reference.
- **NOT Open Source**: This software is **NOT open source** under OSI definitions.
- **Usage Restrictions**: You are strictly prohibited from copying, modifying, distributing, hosting, sublicensing, selling, or deploying this codebase or any portion thereof without explicit prior written consent from the author (**Sudhanshu Batra**).
