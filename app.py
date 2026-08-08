"""
Streamlit Web UI for the RAG System
Provides a clean interface to upload documents and ask questions.

Run with:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

# Suppress verbose transformers warnings in the UI
import logging

from streamlit_mermaid import st_mermaid

from pipeline import RAGPipeline
from utils.helpers import get_pdf_page_image
from utils.logger import setup_logger

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask My Documents — RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .citation-box {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.3rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .chunk-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .score-badge {
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .ml-badge {
        background: #2b1055;
        border: 1px solid #764ba2;
        color: #d8b4fe;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .fallback-warning {
        background: #3d1a1a;
        border: 1px solid #ff4444;
        color: #ff8888;
        padding: 1rem;
        border-radius: 8px;
    }
    .diagram-type-badge {
        background: #238636;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.5rem 2rem;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th {
        background-color: #2b1055 !important;
        color: #d8b4fe !important;
        font-weight: 700;
        text-align: left;
        padding: 10px 14px;
        border: 1px solid #764ba2;
    }
    td {
        padding: 10px 14px;
        border: 1px solid #333;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_mermaid(mermaid_code: str, height: int = 450):
    """Render a Mermaid diagram using the streamlit-mermaid package."""
    st_mermaid(mermaid_code, height=f"{height}px")


def render_citations_with_page_viewer(citations: list[str], retrieved_chunks=None):
    """Render citation cards with expandable visual PDF page previews & document jump targets."""
    with st.expander("📌 Citations & Document Page Viewer", expanded=True):
        for i, cit in enumerate(citations):
            st.markdown(
                f'<div class="citation-box">{cit}</div>',
                unsafe_allow_html=True,
            )
            # Find matching retrieved chunk if available
            rc = (
                retrieved_chunks[i]
                if retrieved_chunks and i < len(retrieved_chunks)
                else None
            )
            if rc and hasattr(rc, "chunk"):
                source = rc.chunk.source
                page = rc.chunk.page
                pdf_path = Path("./tmp_uploads") / source

                if pdf_path.exists() and source.lower().endswith(".pdf"):
                    with st.expander(
                        f"📖 Jump to {source} — Page {page}", expanded=False
                    ):
                        img_bytes = get_pdf_page_image(str(pdf_path), page)
                        if img_bytes:
                            st.image(
                                img_bytes,
                                caption=f"Document Page Snapshot — Page {page} ({source})",
                                use_container_width=True,
                            )
                        st.info(
                            f"**Retrieved Text Segment (Page {page}):**\n\n{rc.chunk.text}"
                        )


# ── Pipeline singleton (cached in session state) ───────────────────────────────
@st.cache_resource(show_spinner="Initializing RAG pipeline…")
def get_pipeline() -> RAGPipeline:
    setup_logger()
    return RAGPipeline()


try:
    pipeline = get_pipeline()
    _pipeline_error = None
except Exception as _e:
    pipeline = None
    _pipeline_error = str(_e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Document Upload")
    st.markdown("Upload one or more documents to add to your knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", use_container_width=True):
            tmp_dir = Path("./tmp_uploads")
            tmp_dir.mkdir(exist_ok=True)

            total_chunks = 0
            for uf in uploaded_files:
                tmp_path = tmp_dir / uf.name
                tmp_path.write_bytes(uf.getvalue())

                with st.spinner(f"Processing {uf.name}…"):
                    try:
                        n = pipeline.ingest(str(tmp_path))
                        total_chunks += n
                        st.success(f"✅ {uf.name}: {n} chunks")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")

            st.info(f"📊 Total chunks indexed: **{total_chunks}**")

    st.divider()
    st.markdown("### 📊 Knowledge Base Stats")
    if pipeline:
        stats = pipeline.get_stats()
        st.metric("Chunks in DB", stats["total_chunks_in_vector_store"])
        st.metric("Embedding Model", stats["embedding_model"].split("/")[-1])

        # ── View & Delete Chunks UI ───────────────────────────────────────────
        with st.expander("🔍 View & Delete Chunks", expanded=False):
            all_chunks = pipeline.get_all_chunks()
            if not all_chunks:
                st.caption("No chunks currently in database.")
            else:
                st.write(f"Showing **{len(all_chunks)}** chunk(s) stored in ChromaDB:")
                for idx, c in enumerate(all_chunks, 1):
                    st.markdown(
                        f"**[{idx}] Source:** `{c.source}` (Page {c.page})  \n"
                        f"`ID: {c.chunk_id[:16]}...`"
                    )
                    st.caption(f"{c.text[:250]}{'…' if len(c.text) > 250 else ''}")
                    if st.button(
                        f"🗑️ Delete Chunk #{idx}",
                        key=f"del_btn_{c.chunk_id}_{idx}",
                        use_container_width=True,
                    ):
                        pipeline.delete_chunk(c.chunk_id)
                        st.success(f"Deleted Chunk #{idx}")
                        st.rerun()
                    st.markdown("---")

                if st.button(
                    "🚨 Reset / Delete All Chunks",
                    key="clear_db_btn",
                    use_container_width=True,
                ):
                    pipeline.reset_database()
                    st.success("Database cleared!")
                    st.rerun()
    else:
        st.error(f"Pipeline error: {_pipeline_error}")

    st.divider()
    st.markdown("### 🤖 ML & RAG Features")
    use_hyde = st.checkbox(
        "🔮 HyDE Retrieval",
        value=False,
        help="Hypothetical Document Embeddings: Generates a sample answer to improve semantic search",
    )
    use_multi_query = st.checkbox(
        "🔀 Multi-Query Expansion",
        value=False,
        help="Generates 2 query variations and merges candidates with RRF",
    )

    st.divider()
    st.markdown("### ⚙️ Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    if debug_mode:
        import logging

        logging.getLogger("rag").setLevel(logging.DEBUG)

    st.divider()
    st.markdown("### 💡 What You Can Do")
    st.markdown("""
    - 🔍 **Search for questions**
    - 📌 **Get citations**
    - 🎨 **Draw flowcharts**
    """)

    with st.expander("⚖️ Why RAG vs. Pasting into ChatGPT", expanded=False):
        st.markdown(
            "Pasting documents directly into ChatGPT/Claude causes two major issues: "
            "**context limits/distraction** (unindexed text degrades model attention) "
            "and **no verifiability** (claims cannot be traced back to exact pages).\n\n"
            "| Dimension | Pasting Docs into ChatGPT / Claude | Production RAG Pipeline |\n"
            "|---|---|---|\n"
            "| **Document Size Limits** | Restricted by model context window; large collections overflow. | Unlimited document corpus scaled across persistent ChromaDB vector store. |\n"
            "| **Source Citations** | None or vague references; cannot verify line or page source. | Enforced `[N]` citations per claim with page numbers & visual page previews. |\n"
            "| **Hallucination Control** | High risk; LLMs guess when facts are missing from prompt. | Low temperature ($0.1$) + prompt guards + automated fallback response. |\n"
            "| **Multi-Doc Search** | Manual copy-pasting & re-formatting of individual files. | Hybrid BM25 (keyword) + Dense Vector (semantic) search across all files. |\n"
            "| **Answer Relevance** | Whole doc dumped as noise; subject to attention degradation. | Cross-Encoder reranking filters out noise, feeding top evidence chunks. |\n"
            "| **Repeatability** | Manual, one-off chat window interaction; non-reusable. | Reusable, production pipeline accessible via Web UI, CLI, and Python API. |\n"
            "| **Evaluation & QA** | No mechanism to measure accuracy or ground truth alignment. | Automated Self-RAG metrics & RAGAS evaluation (faithfulness, relevance). |"
        )

    st.divider()
    st.markdown("*Powered by ChromaDB · Sentence-Transformers · OpenRouter*")


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">📚 Ask My Documents</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Upload documents and ask questions — get answers with citations, or ask to <b>generate diagrams</b> from your content.</div>',
    unsafe_allow_html=True,
)

# ── Prominent RAG vs ChatGPT Comparison Card on Main Site Page ─────────────────
with st.expander(
    "⚖️ Why RAG vs. Pasting Documents into ChatGPT/Claude (Click to View Comparison)",
    expanded=False,
):
    st.markdown(
        "Pasting entire documents directly into a raw LLM prompt (such as ChatGPT or Claude) creates two fundamental failure modes: "
        "**context overflow / distraction** (where high-noise, unindexed text degrades model attention and causes lost-in-the-middle phenomena) "
        "and **lack of verifiability** (where responses cannot be traced back to exact pages or source claims).\n\n"
        "A dedicated RAG architecture solves this by transforming unstructured document collections into an indexed, searchable knowledge base, "
        "retrieving only the highest-relevance evidence chunks, enforcing strict inline citations, and measuring faithfulness quantitatively."
    )
    st.markdown("""
| Dimension | Pasting Docs into ChatGPT / Claude | Production RAG Pipeline |
|---|---|---|
| **Document Size Limits** | Restricted by model context window; large multi-file collections overflow or get truncated. | Unlimited document corpus scaled across persistent ChromaDB vector store. |
| **Source Citations** | None or vague references; cannot verify which line or page generated a statement. | Enforced `[N]` citations per claim with page numbers, text excerpts & visual page previews. |
| **Hallucination Control** | High risk; LLMs guess or improvise when relevant facts are missing from prompt. | Low temperature ($0.1$) + strict prompt guards + automated fallback "insufficient info" response. |
| **Multi-Document Search** | Requires manual copy-pasting and re-formatting of every individual file into prompt window. | Hybrid BM25 (keyword) + Dense Vector (semantic) search across all ingested documents. |
| **Answer Relevance** | Entire document dumped as noise; subject to "lost-in-the-middle" attention degradation. | Cross-Encoder reranking filters out noise, feeding only top-scoring evidence chunks to LLM. |
| **Repeatability** | Manual, one-off chat window interaction with no API, CLI, or programmatic workflow. | Reusable, production pipeline accessible via Streamlit Web UI, CLI, and Python API. |
| **Evaluation & QA** | No mechanism to measure response accuracy or ground truth alignment. | Automated Self-RAG metrics & RAGAS evaluation (faithfulness, correctness, relevance). |
    """)

# Show pipeline error at top if init failed
if pipeline is None:
    st.error(f"⚠️ Pipeline failed to initialize: {_pipeline_error}")
    st.info("Try refreshing the page. If the issue persists, check terminal logs.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_diagram") and msg.get("mermaid_code"):
            st.markdown(
                f'<span class="diagram-type-badge">📊 {msg.get("diagram_type", "diagram")}</span>',
                unsafe_allow_html=True,
            )
            render_mermaid(msg["mermaid_code"])
            with st.expander("</> Mermaid Source"):
                st.code(msg["mermaid_code"], language="text")
        else:
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations_with_page_viewer(msg["citations"], msg.get("chunks"))
            if "chunks" in msg and msg["chunks"] and debug_mode:
                with st.expander("🔍 Retrieved Chunks & ML Metrics (debug)"):
                    if "faithfulness" in msg:
                        st.markdown(
                            f'<span class="ml-badge">🎯 Faithfulness: {msg["faithfulness"]:.2f}</span>'
                            f'<span class="ml-badge">⚡ Context Relevance: {msg.get("relevance", 0.0):.2f}</span>',
                            unsafe_allow_html=True,
                        )
                    for i, rc in enumerate(msg["chunks"], 1):
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<b>[{i}]</b> <span class="score-badge">score: {rc.score:.4f}</span> '
                            f"— <i>{rc.chunk.source}</i>, page {rc.chunk.page}<br><br>"
                            f'{rc.chunk.text[:300]}{"…" if len(rc.chunk.text) > 300 else ""}'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

# Query input
if query := st.chat_input(
    "Ask a question or say 'draw a flowchart of the login process'…"
):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Check if KB has content
    if not pipeline or pipeline.get_stats()["total_chunks_in_vector_store"] == 0:
        warning = "⚠️ No documents ingested yet. Please upload documents using the sidebar first."
        st.session_state.messages.append({"role": "assistant", "content": warning})
        with st.chat_message("assistant"):
            st.warning(warning)

    # ── Diagram Request ───────────────────────────────────────────────────────
    elif pipeline.is_diagram_request(query):
        with st.chat_message("assistant"):
            with st.spinner("🎨 Generating diagram from your documents…"):
                try:
                    start = time.perf_counter()
                    diag = pipeline.generate_diagram(query)
                    elapsed = time.perf_counter() - start

                    if diag.is_fallback:
                        st.markdown(
                            f'<div class="fallback-warning">⚠️ {diag.fallback_message}</div>',
                            unsafe_allow_html=True,
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": diag.fallback_message,
                            }
                        )
                    else:
                        st.markdown(
                            f'<span class="diagram-type-badge">📊 {diag.diagram_type}</span>',
                            unsafe_allow_html=True,
                        )
                        render_mermaid(diag.mermaid_code)
                        with st.expander("</> Mermaid Source Code"):
                            st.code(diag.mermaid_code, language="text")
                        st.caption(
                            f"⚡ Generated in {elapsed:.2f}s | {len(diag.source_chunks)} chunks used"
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"Here is the {diag.diagram_type} diagram:",
                                "is_diagram": True,
                                "mermaid_code": diag.mermaid_code,
                                "diagram_type": diag.diagram_type,
                            }
                        )

                except Exception as e:
                    st.error(f"❌ Diagram generation error: {e}")

    # ── Text Answer ───────────────────────────────────────────────────────────
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                try:
                    start = time.perf_counter()
                    response = pipeline.query(
                        query, use_hyde=use_hyde, use_multi_query=use_multi_query
                    )
                    elapsed = time.perf_counter() - start

                    if response.is_fallback:
                        st.markdown(
                            '<div class="fallback-warning">⚠️ '
                            + response.answer
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(response.answer)

                    # Citations with visual page viewer
                    if response.citations:
                        render_citations_with_page_viewer(
                            response.citations, response.retrieved_chunks
                        )

                    # Debug chunks & ML metrics
                    if debug_mode:
                        with st.expander("🔍 ML Metrics & Retrieved Chunks (debug)"):
                            st.markdown(
                                f'<span class="ml-badge">🎯 Faithfulness: {response.faithfulness_score:.2f}</span>'
                                f'<span class="ml-badge">⚡ Context Relevance: {response.relevance_score:.2f}</span>',
                                unsafe_allow_html=True,
                            )
                            if response.expanded_queries:
                                st.caption(
                                    f"🔀 Expanded Queries: {', '.join(response.expanded_queries)}"
                                )
                            if response.hyde_document:
                                st.caption(
                                    f"🔮 HyDE Passage: {response.hyde_document[:150]}…"
                                )

                            for i, rc in enumerate(response.retrieved_chunks, 1):
                                st.markdown(
                                    f'<div class="chunk-card">'
                                    f'<b>[{i}]</b> <span class="score-badge">score: {rc.score:.4f}</span> '
                                    f"— <i>{rc.chunk.source}</i>, page {rc.chunk.page}<br><br>"
                                    f'{rc.chunk.text[:400]}{"…" if len(rc.chunk.text) > 400 else ""}'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.caption(
                        f"⚡ Answered in {elapsed:.2f}s | {len(response.retrieved_chunks)} chunks retrieved | "
                        f"Faithfulness: {response.faithfulness_score * 100:.0f}%"
                    )

                    # Store in history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "citations": response.citations,
                            "chunks": response.retrieved_chunks,
                            "faithfulness": response.faithfulness_score,
                            "relevance": response.relevance_score,
                        }
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Error: {e}",
                        }
                    )
