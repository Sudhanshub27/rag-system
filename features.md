# 🛠️ Technical Features & System Architecture

A comprehensive technical breakdown of all capabilities in the RAG system, categorized by architectural module.

---

## 1. Ingestion & Document Handling

- **Layout-Aware PDF Ingestion**: Extracts clean per-page text from PDF files using `PyMuPDF` (`fitz`), preserving structural page metadata (`page`, `total_pages`).
- **Hyperlink & URI Extraction**: Automatically detects and extracts embedded hyperlink URIs and anchor texts from PDF annotations using `PyMuPDF` page object parsing.
- **Robust PDF Fallback Engine**: Seamlessly falls back to `pypdf` (`PdfReader`) if PyMuPDF is not installed, preserving annotations and document structure.
- **Multi-Format Text Ingestion**: Reads UTF-8 encoded plaintext (`.txt`) and Markdown (`.md`) documents with unicode normalization (`unicodedata`).
- **Automated Pipeline Routing**: `IngestionPipeline` inspects file extensions and dispatches documents to specialized loaders automatically.
- **Directory Ingestion**: Recursively discovers, ingests, and indexes entire document directories in a single command.

---

## 2. Chunking Strategy

- **Sentence-Boundary Aware Splitting**: `SemanticChunker` uses regular expression heuristics (`re`) to split text cleanly on sentence boundaries rather than arbitrary character counts.
- **Pitch-Deck Optimized Tokens**: Configurable target chunk budget (default: `250` tokens) with overlap (default: `50` tokens) tailored for short pitch deck slides and dense documentation.
- **Minimum Token Guard (No Slide Left Behind)**: `min_chunk_size` set to `15` tokens to prevent concise pitch deck bullet points, tables, and short slides from being discarded during ingestion.
- **Hard-Splitting Long Sentences**: Word-level fallback splitting handles oversized paragraphs or unpunctuated text blocks that exceed maximum chunk budgets.
- **Deterministic Chunk Identifiers**: Generates stable, reproducible 16-character SHA-256 hashes (`hashlib`) based on `source`, `chunk_index`, and content snippet.

---

## 3. Embeddings & Vector Storage

- **Dense Sentence Embeddings**: Encodes document chunks into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
- **Embedding Cache Engine**: Hashes chunk text (`MD5`) and caches vector output on disk (`embeddings_cache.pkl`) to eliminate duplicate model inferences.
- **Persistent ChromaDB Vector Index**: Stores document vectors and rich metadata (`source`, `page`, `chunk_index`) in a persistent local `ChromaDB` collection.
- **Cosine Distance Search**: Performs fast $k$-nearest neighbor similarity lookups using ChromaDB's native cosine distance metric.

---

## 4. Retrieval (Hybrid Search & Reranking)

- **BM25 Lexical Keyword Search**: Performs sparse keyword matching powered by `rank_bm25` (`BM25Okapi`), preserving exact terms, acronyms, and proper nouns.
- **Reciprocal Rank Fusion (RRF)**: Merges BM25 keyword rankings and vector similarity rankings using reciprocal rank scoring ($RRF\_Score = \frac{w_{bm25}}{60 + k} + \frac{w_{vector}}{60 + k}$).
- **Cross-Encoder Reranking**: Re-evaluates top retrieved candidate pairs using `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2` for maximum precision.
- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer passage using the LLM to bridge semantic vocabulary gaps between user queries and document content.
- **Multi-Query Expansion**: Formulates multiple query reformulations using the LLM and merges unique candidate chunks via RRF fusion.
- **Broad Query Context Expansion**: Automatically detects overview/summary queries (*"explain pitch deck"*, *"summarize document"*) and expands retrieval budget to 8 full chunks.

---

## 5. Generation & Hallucination Prevention

- **Multi-LLM Provider Support**: Connects seamlessly to OpenRouter (`openrouter/free`), DeepSeek (`deepseek-chat`), Google Gemini (`gemini-2.0-flash`), Anthropic Claude (`claude-3-5-sonnet`), or OpenAI (`gpt-4o-mini`).
- **Citation Enforcement Prompting**: Instructs the LLM via versioned system prompts (`config/prompts.yaml`) to attach `[N]` inline citations for every factual statement.
- **Hallucination Guardrails & Fallback**: Operates at low temperature ($0.1$) and returns an automated fallback response when retrieved context is empty or irrelevant.
- **Mermaid Diagram Generation**: Detects diagram requests (*"draw a flowchart"*, *"sequence diagram"*) and outputs valid Mermaid.js syntax.

---

## 6. Evaluation & CI/CD

- **RAGAS Evaluation Framework**: Evaluates overall pipeline accuracy using `RAGAS` metrics (Faithfulness, Answer Correctness, Context Relevance) against a golden dataset (`golden_dataset.json`).
- **Self-RAG Quantitative Scoring**: Computes real-time Faithfulness (grounding ratio) and Context Relevance scores for every live answer generated.
- **Automated CI/CD Pipeline**: GitHub Actions workflow (`ci.yml`) runs formatting checks (`black`), linting (`ruff`), and unit/integration test suite (`pytest`) on every commit.
- **Quality Gate Threshold Enforcement**: `evaluate.py` option `--fail-on-threshold` blocks build deployments if metrics drop below configurable precision targets.

---

## 7. Interfaces (Web UI & CLI)

- **Interactive Streamlit Web UI**: Full-featured web interface (`app.py`) for document uploads, interactive Q&A, and real-time metric display.
- **Visual PDF Page Previewer**: Renders crisp visual snapshots of exact PDF document pages using `PyMuPDF` when expanding citation cards.
- **Interactive Mermaid Diagram Renderer**: Renders interactive flowcharts and sequence diagrams directly in the browser via `streamlit-mermaid`.
- **Knowledge Base Management UI**: Inspection tools in Streamlit sidebar to view stored chunks, delete individual chunk IDs, or purge the ChromaDB database.
- **Command-Line Interface (CLI)**: Full terminal control (`cli.py`) powered by `argparse` supporting single/directory ingestion, query execution with `--json` output, and database stats.
