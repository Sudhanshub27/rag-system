"""
Streamlit Web UI for the RAG System
Modern, Chat-Bubble Interface with Split-View Source Inspector & PDF Page Previews.

Run with:
    streamlit run app.py
"""

import logging
import sys
import time
from pathlib import Path

import streamlit as st
from streamlit_mermaid import st_mermaid

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import RAGPipeline
from utils.helpers import get_pdf_page_image
from utils.logger import setup_logger

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask My Documents — RAG Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Modern CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Dark Glassmorphism Design Token System */
    .stApp {
        background-color: #0b0f19;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* User & Assistant Chat Bubbles */
    .user-bubble-container {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
    }
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        padding: 0.9rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 85%;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .assistant-bubble-container {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
    }
    .assistant-card {
        background: #1e293b;
        border: 1px solid #334155;
        color: #f8fafc;
        padding: 1.2rem 1.4rem;
        border-radius: 4px 18px 18px 18px;
        width: 100%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Citation Pill Badges */
    .citation-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid #6366f1;
        color: #a5b4fc;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 12px;
        margin: 0 3px;
        font-family: monospace;
    }

    /* Source Inspector Panel */
    .inspector-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        margin-top: 1rem;
    }
    .inspector-title {
        color: #c084fc;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.5rem;
    }

    /* Stat Metrics Cards */
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: #818cf8;
    }
    .stat-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Score Chips */
    .score-chip {
        background: #1e1b4b;
        border: 1px solid #4338ca;
        color: #a5b4fc;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .ml-chip {
        background: #2b1055;
        border: 1px solid #764ba2;
        color: #d8b4fe;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
    }

    /* Styled Markdown Table */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th {
        background-color: #1e1b4b !important;
        color: #c084fc !important;
        font-weight: 700;
        text-align: left;
        padding: 10px 14px;
        border: 1px solid #4338ca;
    }
    td {
        padding: 10px 14px;
        border: 1px solid #334155;
        font-size: 0.9rem;
    }

    /* Custom Streamlit Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_mermaid(mermaid_code: str, height: int = 450):
    """Render a Mermaid diagram using the streamlit-mermaid package."""
    st_mermaid(mermaid_code, height=f"{height}px")


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

# ── Session State Initialization ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_chunk" not in st.session_state:
    st.session_state.selected_chunk = None
if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None


def get_ingested_docs_summary(p):
    """Group chunks by document source."""
    if not p:
        return []
    chunks = p.get_all_chunks()
    doc_map = {}
    for c in chunks:
        src = c.source
        if src not in doc_map:
            doc_map[src] = {"source": src, "chunk_count": 0}
        doc_map[src]["chunk_count"] += 1
    return list(doc_map.values())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Document Upload")
    st.markdown("Upload documents to build your knowledge base.")

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

    # Knowledge Base Stats & File Management
    st.markdown("### 📊 Knowledge Base Stats")
    if pipeline:
        stats = pipeline.get_stats()
        doc_summary = get_ingested_docs_summary(pipeline)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-val">{stats.get("total_chunks_in_vector_store", 0)}</div><div class="stat-lbl">Total Chunks</div></div>',
                unsafe_allow_html=True,
            )
        with col_s2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-val">{len(doc_summary)}</div><div class="stat-lbl">Documents</div></div>',
                unsafe_allow_html=True,
            )

        # Ingested Files List with Delete Action
        if doc_summary:
            st.markdown("##### 📄 Ingested Files")
            for d in doc_summary:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**`{d['source']}`**  \n*{d['chunk_count']} chunks*")
                with c2:
                    if st.button(
                        "🗑️",
                        key=f"del_doc_{d['source']}",
                        help=f"Delete {d['source']}",
                    ):
                        pipeline.delete_document(d["source"])
                        st.success(f"Deleted {d['source']}")
                        st.rerun()

        # Collapsible Database Inspector & Reset
        with st.expander("🔍 View & Delete Chunks", expanded=False):
            all_chunks = pipeline.get_all_chunks()
            if not all_chunks:
                st.caption("No chunks currently in database.")
            else:
                st.write(f"Stored Chunks: **{len(all_chunks)}**")
                for idx, c in enumerate(all_chunks, 1):
                    st.markdown(
                        f"**[{idx}] Source:** `{c.source}` (Page {c.page})  \n`ID: {c.chunk_id[:16]}...`"
                    )
                    st.caption(f"{c.text[:200]}{'…' if len(c.text) > 200 else ''}")
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
                    "🚨 Reset Database",
                    key="clear_db_btn",
                    use_container_width=True,
                ):
                    pipeline.reset_database()
                    st.success("Database cleared!")
                    st.rerun()
    else:
        st.error(f"Pipeline error: {_pipeline_error}")

    st.divider()

    # Layout & Feature Toggles
    st.markdown("### ⚙️ View & ML Features")
    split_view = st.checkbox("📑 Split View (Inspector Panel)", value=True)
    debug_mode = st.checkbox("🛠️ Debug Mode & Retrieval Scores", value=False)
    use_hyde = st.checkbox("🔮 HyDE Retrieval", value=False)
    use_multi_query = st.checkbox("🔀 Multi-Query Expansion", value=False)

    if debug_mode:
        import logging

        logging.getLogger("rag").setLevel(logging.DEBUG)

    st.divider()
    st.markdown("*Powered by ChromaDB · Sentence-Transformers · OpenRouter*")


# ── Main Layout (Split View or Single Column) ─────────────────────────────────
if split_view:
    col_chat, col_inspector = st.columns([3, 2])
else:
    col_chat = st.container()
    col_inspector = None

with col_chat:
    st.markdown(
        '<div class="main-header">📚 Ask My Documents</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Upload documents and ask questions — get grounded answers with inline citations & page previews.</div>',
        unsafe_allow_html=True,
    )

    # ── Prominent RAG vs ChatGPT Comparison Card ─────────────────────────────
    with st.expander(
        "⚖️ Why RAG vs. Pasting Documents into ChatGPT/Claude (Architecture Comparison)",
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

    # Show pipeline error if init failed
    if pipeline is None:
        st.error(f"⚠️ Pipeline failed to initialize: {_pipeline_error}")
        st.stop()

    # ── Chat Messages Display ────────────────────────────────────────────────
    for msg_idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]

        # User Message (Right Aligned Bubble)
        if role == "user":
            st.markdown(
                f'<div class="user-bubble-container"><div class="user-bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )

        # Assistant Message (Left Aligned Card)
        else:
            with st.container():
                st.markdown(
                    f'<div class="assistant-bubble-container"><div class="assistant-card">🤖 <b>Assistant</b><br><br>{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )

                # Render Citations & Interactive Chunk Selectors
                if msg.get("citations"):
                    chunks = msg.get("chunks", [])
                    st.markdown("##### 📌 Citations & Source Chunks")

                    cols = st.columns(min(len(msg["citations"]), 4))
                    for idx, cit in enumerate(msg["citations"]):
                        c_col = cols[idx % len(cols)]
                        rc = chunks[idx] if idx < len(chunks) else None
                        source_name = rc.chunk.source if rc else f"Source {idx+1}"
                        page_num = rc.chunk.page if rc else 1

                        with c_col:
                            if st.button(
                                f"📄 [{idx+1}] {source_name[:15]}… p.{page_num}",
                                key=f"cit_btn_{msg_idx}_{idx}",
                                help=f"Click to inspect Page {page_num} of {source_name} in Split View",
                            ):
                                st.session_state.selected_chunk = rc
                                st.session_state.selected_citation = cit
                                if not split_view:
                                    st.info(
                                        f"Selected [{idx+1}]: Page {page_num} of {source_name}"
                                    )
                                st.rerun()

                    # Expandable Page Snapshot Previews
                    with st.expander(
                        "📄 View Page Snapshots & Excerpts", expanded=False
                    ):
                        for idx, cit in enumerate(msg["citations"]):
                            rc = chunks[idx] if idx < len(chunks) else None
                            if rc and hasattr(rc, "chunk"):
                                src = rc.chunk.source
                                pg = rc.chunk.page
                                pdf_path = Path("./tmp_uploads") / src
                                st.markdown(f"**[{idx+1}] `{src}` — Page {pg}**")
                                if pdf_path.exists() and src.lower().endswith(".pdf"):
                                    img_bytes = get_pdf_page_image(str(pdf_path), pg)
                                    if img_bytes:
                                        st.image(
                                            img_bytes,
                                            caption=f"Snapshot Page {pg} ({src})",
                                            use_container_width=True,
                                        )
                                st.info(f"**Text Excerpt:**\n> {rc.chunk.text}")
                                st.markdown("---")

                # Debug Metrics & Retrieval Scores
                if debug_mode and msg.get("chunks"):
                    with st.expander(
                        "🛠️ Retrieval Scores & ML Metrics (Debug)",
                        expanded=False,
                    ):
                        if "faithfulness" in msg:
                            st.markdown(
                                f'<span class="ml-chip">🎯 Faithfulness: {msg["faithfulness"]:.2f}</span>'
                                f'<span class="ml-chip">⚡ Relevance: {msg.get("relevance", 0.0):.2f}</span>',
                                unsafe_allow_html=True,
                            )
                        for i, rc in enumerate(msg["chunks"], 1):
                            st.markdown(
                                f'<div class="score-chip">Rerank Score: {rc.score:.4f}</div>'
                                f"<b>[{i}]</b> <i>{rc.chunk.source}</i> (Page {rc.chunk.page})<br>"
                                f"<code>{rc.chunk.text[:250]}…</code>",
                                unsafe_allow_html=True,
                            )

    # ── Query Input ───────────────────────────────────────────────────────────
    if query := st.chat_input(
        "Ask a question or say 'draw a flowchart of the login process'…"
    ):
        st.session_state.messages.append({"role": "user", "content": query})

        if not pipeline or pipeline.get_stats()["total_chunks_in_vector_store"] == 0:
            warning = "⚠️ No documents ingested yet. Please upload documents in the sidebar first."
            st.session_state.messages.append({"role": "assistant", "content": warning})
            st.rerun()

        elif pipeline.is_diagram_request(query):
            with st.spinner("🎨 Generating diagram from documents…"):
                diag = pipeline.generate_diagram(query)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Here is the requested **{diag.diagram_type}** diagram:",
                        "is_diagram": True,
                        "mermaid_code": diag.mermaid_code,
                        "diagram_type": diag.diagram_type,
                    }
                )
                st.rerun()

        else:
            with st.spinner("Searching documents & reranking evidence…"):
                start = time.perf_counter()
                response = pipeline.query(
                    query, use_hyde=use_hyde, use_multi_query=use_multi_query
                )
                elapsed = time.perf_counter() - start

                # Select top chunk by default into Split Inspector
                if response.retrieved_chunks:
                    st.session_state.selected_chunk = response.retrieved_chunks[0]
                    if response.citations:
                        st.session_state.selected_citation = response.citations[0]

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "citations": response.citations,
                        "chunks": response.retrieved_chunks,
                        "faithfulness": response.faithfulness_score,
                        "relevance": response.relevance_score,
                        "elapsed": elapsed,
                    }
                )
                st.rerun()

# ── Right Panel: Source Inspector (Split View) ───────────────────────────────
if split_view and col_inspector:
    with col_inspector:
        st.markdown(
            '<div class="inspector-title">📑 Source Document Inspector</div>',
            unsafe_allow_html=True,
        )

        selected_rc = st.session_state.selected_chunk

        if selected_rc and hasattr(selected_rc, "chunk"):
            chunk = selected_rc.chunk
            score = getattr(selected_rc, "score", 0.0)
            rank = getattr(selected_rc, "rank", 1)

            st.markdown(
                f'<div class="inspector-card">'
                f"<h4>📄 {chunk.source}</h4>"
                f'<div class="score-chip">Page {chunk.page}</div>'
                f'<div class="score-chip">Rerank Score: {score:.4f}</div>'
                f'<div class="score-chip">Chunk ID: {chunk.chunk_id[:12]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("##### 📝 Text Content Chunk")
            st.info(chunk.text)

            # High-Resolution PDF Page Snapshot Preview
            pdf_path = Path("./tmp_uploads") / chunk.source
            if pdf_path.exists() and chunk.source.lower().endswith(".pdf"):
                st.markdown(f"##### 📸 PDF Page Snapshot (Page {chunk.page})")
                img_bytes = get_pdf_page_image(str(pdf_path), chunk.page)
                if img_bytes:
                    st.image(
                        img_bytes,
                        caption=f"Visual Page Snapshot — {chunk.source} (Page {chunk.page})",
                        use_container_width=True,
                    )
            elif not pdf_path.exists():
                st.caption("💡 Upload file in sidebar to see visual PDF snapshots.")

        else:
            st.info(
                "💡 **No citation selected yet.**  \n"
                "Ask a question and click any citation badge **`[1]`**, **`[2]`** to inspect its exact text chunk & visual PDF page snapshot here."
            )
