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

from pipeline import RAGPipeline
from utils.logger import setup_logger
from streamlit_mermaid import st_mermaid

# Suppress verbose transformers warnings in the UI
import logging
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
st.markdown("""
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
    .fallback-warning {
        background: #3d1a1a;
        border: 1px solid #ff4444;
        color: #ff8888;
        padding: 1rem;
        border-radius: 8px;
    }
    .diagram-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
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
</style>
""", unsafe_allow_html=True)


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
    else:
        st.error(f"Pipeline error: {_pipeline_error}")

    st.divider()
    st.markdown("### ⚙️ Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    if debug_mode:
        import logging
        logging.getLogger("rag").setLevel(logging.DEBUG)

    st.divider()
    st.markdown("### 📊 Diagram Types")
    st.markdown("""
    Ask for diagrams using natural language:
    - `draw a flowchart of...`
    - `create a class diagram for...`
    - `show sequence diagram of...`
    - `make an ER diagram of...`
    - `mind map of...`
    """)

    st.divider()
    st.markdown("*Powered by ChromaDB · Sentence-Transformers · OpenRouter*")


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📚 Ask My Documents</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload documents and ask questions — get answers with citations, or ask to <b>generate diagrams</b> from your content.</div>', unsafe_allow_html=True)

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
            if "citations" in msg and msg["citations"]:
                with st.expander("📌 Citations"):
                    for cit in msg["citations"]:
                        st.markdown(f'<div class="citation-box">{cit}</div>', unsafe_allow_html=True)
            if "chunks" in msg and msg["chunks"] and debug_mode:
                with st.expander("🔍 Retrieved Chunks (debug)"):
                    for i, rc in enumerate(msg["chunks"], 1):
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<b>[{i}]</b> <span class="score-badge">score: {rc.score:.4f}</span> '
                            f'— <i>{rc.chunk.source}</i>, page {rc.chunk.page}<br><br>'
                            f'{rc.chunk.text[:300]}{"…" if len(rc.chunk.text) > 300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# Query input
if query := st.chat_input("Ask a question or say 'draw a flowchart of the login process'…"):
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
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": diag.fallback_message,
                        })
                    else:
                        st.markdown(
                            f'<span class="diagram-type-badge">📊 {diag.diagram_type}</span>',
                            unsafe_allow_html=True,
                        )
                        render_mermaid(diag.mermaid_code)
                        with st.expander("</> Mermaid Source Code"):
                            st.code(diag.mermaid_code, language="text")
                        st.caption(f"⚡ Generated in {elapsed:.2f}s | {len(diag.source_chunks)} chunks used")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Here is the {diag.diagram_type} diagram:",
                            "is_diagram": True,
                            "mermaid_code": diag.mermaid_code,
                            "diagram_type": diag.diagram_type,
                        })

                except Exception as e:
                    st.error(f"❌ Diagram generation error: {e}")

    # ── Text Answer ───────────────────────────────────────────────────────────
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                try:
                    start = time.perf_counter()
                    response = pipeline.query(query)
                    elapsed = time.perf_counter() - start

                    if response.is_fallback:
                        st.markdown(
                            '<div class="fallback-warning">⚠️ '
                            + response.answer
                            + '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(response.answer)

                    # Citations
                    if response.citations:
                        with st.expander("📌 Citations", expanded=True):
                            for cit in response.citations:
                                st.markdown(
                                    f'<div class="citation-box">{cit}</div>',
                                    unsafe_allow_html=True,
                                )

                    # Debug chunks
                    if debug_mode and response.retrieved_chunks:
                        with st.expander("🔍 Retrieved Chunks (debug)"):
                            for i, rc in enumerate(response.retrieved_chunks, 1):
                                st.markdown(
                                    f'<div class="chunk-card">'
                                    f'<b>[{i}]</b> <span class="score-badge">score: {rc.score:.4f}</span> '
                                    f'— <i>{rc.chunk.source}</i>, page {rc.chunk.page}<br><br>'
                                    f'{rc.chunk.text[:400]}{"…" if len(rc.chunk.text) > 400 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.caption(f"⚡ Answered in {elapsed:.2f}s | {len(response.retrieved_chunks)} chunks retrieved")

                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "citations": response.citations,
                        "chunks": response.retrieved_chunks,
                    })

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                    })

