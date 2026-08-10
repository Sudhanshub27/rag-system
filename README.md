# 📚 Ask My Documents — Production RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Evaluation](https://img.shields.io/badge/eval-RAGAS-green.svg)](evaluation/)
[![Lint](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml/badge.svg?job=lint)](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml)
[![Tests](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml/badge.svg?job=test)](https://github.com/Sudhanshub27/rag-system/actions/workflows/ci.yml)

A **production-grade Retrieval-Augmented Generation (RAG)** system equipped with **HyDE**, **Multi-Query Expansion**, **Self-RAG Evaluation**, **Mermaid Diagram Generation**, and **Hyperlink-Aware Ingestion**. All answers are strictly grounded in evidence with rich inline text quote citations.

---

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🔓 No Login Required](#-no-login-required)
- [⚖️ Why RAG vs. Pasting Docs into ChatGPT/Claude](#-why-rag-vs-pasting-documents-into-chatgptclaude)
- [🧩 Retrieval Modes & ML Features](#-retrieval-modes--ml-features)
- [💬 Interaction & Application Modes](#-interaction--application-modes)
- [📊 Self-RAG & Evaluation Metrics](#-self-rag--evaluation-metrics)
- [⚡ Quick Start & Running Locally](#-quick-start--running-locally)
- [☁️ Streamlit Cloud Deployment & Secrets Setup](#️-streamlit-cloud-deployment--secrets-setup)
- [⚙️ Configuration & Tuning](#️-configuration--tuning)
- [🧪 Testing & CI Compliance](#-testing--ci-compliance)
- [🔐 Multi-LLM Provider Support](#-multi-llm-provider-support)

---

## ✨ Key Features

| Category | Capability & Technology Stack | Technical Details |
|---|---|---|
| 📂 **1. Ingestion & Docs** | Layout-Aware & Hyperlink PDF Ingestion | Extracts per-page text & embedded hyperlinks via `PyMuPDF` (`fitz`), with `pypdf` fallback. |
| ✂️ **2. Chunking Strategy** | Pitch-Deck Optimized Semantic Chunker | Sentence-boundary aware regex splitting (`250` token size, `15` token min limit to preserve concise slides). |
| 🗄️ **3. Embeddings & Storage** | Dense Vectors & Persistent ChromaDB | `SentenceTransformers all-MiniLM-L6-v2` (384-dim) with disk-cached hash lookups (`pkl`). |
| 🔍 **4. Retrieval Engine** | Hybrid BM25 + Vector Search + Reranker | Combines `rank_bm25` and ChromaDB via RRF fusion + `ms-marco-MiniLM-L-6-v2` cross-encoder reranking. |
| 🤖 **5. Generation & Guards** | Multi-LLM & Hallucination Prevention | Supports OpenRouter / Claude / DeepSeek / Gemini / OpenAI at temp `0.1` with enforced `[N]` citations & fallback response. |
| 📊 **6. Evaluation & CI/CD** | Self-RAG Scoring & RAGAS Framework | Computes real-time Faithfulness/Relevance metrics; automated `pytest` and `ruff` GitHub Actions CI gates. |
| 🖥️ **7. User Interfaces** | Interactive Streamlit Web UI & CLI | Web UI (`app.py`) with visual PDF page snapshots & Mermaid diagrams; full terminal CLI (`cli.py`). |

---

## 🔓 No Login Required

This application is engineered with a **zero-friction, privacy-preserving multi-tenant architecture**. You can upload and query documents instantly without creating an account:

- **Zero Authentication Overhead**: No email, username, password, or signup process required.
- **Anonymous Cookie-Based Persistence**: On your first visit, the system generates a private, random UUID4 identifier (`tenant_id`) stored in a long-lived browser cookie (2-year expiry). Returning to the app on the same browser automatically restores your private workspace.
- **Strict Data Isolation**: Documents, vector embeddings, and BM25 index data are partitioned into isolated per-tenant vector store collections (`tenant_<tenant_id>`). No user can query or view another user's content.
- **Persistent Workspace**: Your uploaded documents and index data remain saved across browser sessions (days, weeks, or months) as long as the cookie isn't cleared.
- **User-Controlled Data Purging**: Permanently delete all your uploaded documents and vector index collections at any time using the **"Delete all my data"** button in the sidebar.
- **Privacy Limitation & Note**: Because there is no account system linking personal identity to your data, **clearing browser cookies or switching devices/browsers assigns a new anonymous tenant ID**. Access to documents uploaded under the old browser cookie cannot be recovered once the cookie is lost.

---


## ⚖️ Why RAG vs. Pasting Documents into ChatGPT/Claude

Pasting entire documents directly into a raw LLM prompt (such as ChatGPT or Claude) creates two fundamental failure modes: **context overflow / distraction** (where high-noise, unindexed text degrades model attention and causes lost-in-the-middle phenomena) and **lack of verifiability** (where responses cannot be traced back to exact pages or source claims). A dedicated RAG architecture solves this by transforming unstructured document collections into an indexed, searchable knowledge base, retrieving only the highest-relevance evidence chunks, enforcing strict inline citations, and measuring faithfulness quantitatively.

| Dimension | Pasting Docs into ChatGPT / Claude | Production RAG Pipeline |
|---|---|---|
| **Document Size Limits** | Restricted by model context window; large multi-file collections overflow or get truncated. | Unlimited document corpus scaled across persistent ChromaDB vector store. |
| **Source Citations** | None or vague references; cannot verify which line or page generated a statement. | Enforced `[N]` citations per claim with page numbers, excerpts, and visual page previews. |
| **Hallucination Control** | High risk; LLMs guess or improvise when relevant facts are missing from prompt. | Low temperature ($0.1$) + strict prompt guards + automated fallback "insufficient info" response. |
| **Multi-Document Search** | Requires manual copy-pasting and re-formatting of every individual file into prompt window. | Hybrid BM25 (keyword) + Dense Vector (semantic) search across all ingested documents. |
| **Answer Relevance** | Entire document dumped as noise; subject to "lost-in-the-middle" attention degradation. | Cross-Encoder reranking filters out noise, feeding only top-scoring evidence chunks to LLM. |
| **Repeatability** | Manual, one-off chat window interaction with no API, CLI, or programmatic workflow. | Reusable, production pipeline accessible via Streamlit Web UI, CLI, and Python API. |
| **Evaluation & QA** | No mechanism to measure response accuracy or ground truth alignment. | Automated Self-RAG metrics & RAGAS evaluation (faithfulness, correctness, relevance). |

---

## 🧩 Retrieval Modes & ML Features

The system supports multiple advanced retrieval strategies configured via the sidebar **🤖 ML & RAG Features** panel:

### 1. Hybrid Retrieval (Default: BM25 + Vector Search + RRF + Cross-Encoder)
- **BM25 Keyword Search**: Captures exact terminology, product codes, proper nouns, and numbers.
- **Dense Vector Search**: Captures semantic intent using SentenceTransformer embeddings.
- **Reciprocal Rank Fusion (RRF)**: Merges keyword and vector rankings into a unified score using $RRF\_Score = \frac{w_{bm25}}{60 + k} + \frac{w_{vector}}{60 + k}$.
- **Cross-Encoder Reranking**: Re-evaluates top candidate pairs using `ms-marco-MiniLM-L-6-v2` for high precision.

### 2. HyDE (Hypothetical Document Embeddings)
- **What it does**: When enabled, the LLM first generates a *hypothetical target document* that would answer the user's prompt.
- **Why it helps**: User queries (e.g., *"How do I reset password?"*) often have poor vector overlap with document text (*"Account authentication recovery protocols..."*). Embedding the hypothetical answer bridges the semantic gap.

### 3. Multi-Query Expansion Mode
- **What it does**: Automatically reformulates the user prompt into multiple semantic variations (e.g., alternative phrasing, synomyms).
- **Why it helps**: Queries vector index for each variation and merges unique chunks, maximizing retrieval recall for complex or ambiguous questions.

### 4. Pitch Deck & Summary Mode
- **What it does**: Automatically detects broad query patterns like *"explain the pitch deck"* or *"summarize document"*.
- **Why it helps**: Expands context retrieval to **up to 8 full chunks** so no slide (market, financials, team, business model) is missed.

---

## 💬 Interaction & Application Modes

### 1. Q&A Mode with Quote Citations
Ask any natural language question about your uploaded documents. Answers are citation-grounded with clickable quote cards showing the exact page number and text snippet.

### 2. Diagram & Flowchart Generation Mode
To generate visual flowcharts or sequence diagrams, simply phrase your query with diagram keywords:
- *"draw a flowchart of the login process"*
- *"generate a sequence diagram for payment processing"*
- *"create a flowchart showing user onboarding"*

The LLM outputs clean Mermaid.js syntax that renders interactive flowcharts directly in the UI.

### 3. Knowledge Base Inspection & Maintenance Mode
Under the **View & Delete Chunks** section in the sidebar:
- **Inspect Chunks**: View chunk IDs, exact text content, and token counts.
- **Granular Delete**: Delete individual chunks or remove entire files.
- **Reset Database**: Purge the index with one click.

---

## 📊 Self-RAG & Evaluation Metrics

Every response evaluates its own quality in real time:

- **Faithfulness Score ($0.0 - 1.0$)**: Measures what percentage of key claims in the generated answer are grounded in the retrieved chunks.
- **Context Relevance Score ($0.0 - 1.0$)**: Measures how relevant the retrieved context chunks are to the user's question.

*Enable **Debug Mode** in the UI to view real-time Faithfulness/Relevance scores, HyDE drafts, and query expansion variations.*

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

### 2. Environment Variables

Copy `.env.example` to `.env` and add your preferred API key:

```bash
cp .env.example .env
```

Example `.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key
# Optional alternatives:
# ANTHROPIC_API_KEY=sk-ant-xxx
# OPENAI_API_KEY=sk-xxx
# DEEPSEEK_API_KEY=sk-xxx
# GEMINI_API_KEY=AIzaSy...
```

### 3. Launch Application

```bash
./run.sh
```
or
```bash
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment & Secrets Setup

When deploying to **Streamlit Community Cloud** (`share.streamlit.io`):

1. Go to your Streamlit Cloud Dashboard $\rightarrow$ Click **Manage app** $\rightarrow$ **Settings** $\rightarrow$ **Secrets**.
2. Enter your API key in **TOML format (with quotes)**:

```toml
OPENROUTER_API_KEY = "sk-or-v1-your-actual-openrouter-key-here"

# Optional alternatives:
DEEPSEEK_API_KEY = "sk-0ec8368be..."
GEMINI_API_KEY = "AQ.AbRN6K4..."
OPENAI_API_KEY = "sk-proj-..."
```

3. Click **Save changes**. The system auto-detects `st.secrets` seamlessly.

---

## ⚙️ Configuration & Tuning

Key parameters in `config/settings.yaml`:

```yaml
chunking:
  chunk_size: 250          # Tokens per chunk (optimal for slides & standard documents)
  chunk_overlap: 50        # Token overlap
  min_chunk_size: 15       # Discard limit (preserves short bullet points & tables)

retrieval:
  top_k: 12                # Candidate chunks retrieved from vector store
  top_n_rerank: 6          # Final chunks retained after cross-encoder reranking
  use_bm25: true           # Enable BM25 keyword search
  bm25_weight: 0.3         # Fusion weight
  vector_weight: 0.7       # Fusion weight
```

---

## 🧪 Testing & CI Compliance

Run the automated test suite (31 unit & integration tests):

```bash
# Run pytest test suite
.venv/bin/python -m pytest

# Run linter and formatting checks
.venv/bin/ruff check .
.venv/bin/black --check .
```

---

## 🔐 Multi-LLM Provider Support

The system supports multiple LLM providers out-of-the-box:
- **OpenRouter** (`openrouter/free` auto-routed model)
- **DeepSeek** (`deepseek-chat`)
- **Google Gemini** (`gemini-2.0-flash`)
- **Anthropic Claude** (`claude-3-5-sonnet`)
- **OpenAI** (`gpt-4o-mini`)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
