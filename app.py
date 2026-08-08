"""
Streamlit Web UI for the RAG System
Minimal, Professional Interface (Linear/Perplexity Aesthetic) with Multi-Tenant Isolation.

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
from utils.auth import authenticate_user, register_user
from utils.helpers import get_pdf_page_image
from utils.logger import setup_logger

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence — RAG System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS Design Token System (Linear / Perplexity Palette) ───────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: #0B0E14;
        color: #E6E8EB;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B0E14;
        border-right: 1px solid #1F2430;
    }

    /* Header Typography */
    .main-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #E6E8EB;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 0.875rem;
        color: #8B92A3;
        margin-bottom: 1.5rem;
    }

    /* Message Containers */
    .user-card {
        background-color: #1A1F2E;
        border-left: 3px solid #6366F1;
        border-top: 1px solid #1F2430;
        border-right: 1px solid #1F2430;
        border-bottom: 1px solid #1F2430;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #E6E8EB;
        font-size: 0.9375rem;
        line-height: 1.6;
    }

    .assistant-card {
        background-color: #12161F;
        border: 1px solid #1F2430;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        color: #E6E8EB;
        font-size: 0.9375rem;
        line-height: 1.6;
    }

    .role-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8B92A3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* Citation Badges */
    .citation-pill {
        display: inline-flex;
        align-items: center;
        background-color: #12161F;
        border: 1px solid #1F2430;
        color: #6366F1;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin: 0.2rem;
        font-family: monospace;
    }

    /* Inspector Panel */
    .inspector-box {
        background-color: #12161F;
        border: 1px solid #1F2430;
        border-radius: 6px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    .inspector-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: #8B92A3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #1F2430;
        padding-bottom: 0.5rem;
    }

    /* Stat Cards */
    .stat-card {
        background-color: #12161F;
        border: 1px solid #1F2430;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6366F1;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #8B92A3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Score Chips */
    .score-chip {
        background-color: #1A1F2E;
        border: 1px solid #1F2430;
        color: #8B92A3;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Empty State Card */
    .empty-state-box {
        background-color: #12161F;
        border: 1px dashed #1F2430;
        border-radius: 8px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: #8B92A3;
        margin: 2rem 0;
    }
    .empty-state-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E6E8EB;
        margin-bottom: 0.5rem;
    }

    /* Table Styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th {
        background-color: #12161F !important;
        color: #E6E8EB !important;
        font-weight: 600;
        text-align: left;
        padding: 8px 12px;
        border: 1px solid #1F2430;
        font-size: 0.85rem;
    }
    td {
        padding: 8px 12px;
        border: 1px solid #1F2430;
        font-size: 0.85rem;
        color: #8B92A3;
    }

    /* Custom Streamlit Buttons */
    .stButton>button {
        background-color: #6366F1;
        color: #FFFFFF;
        border: none;
        font-weight: 500;
        font-size: 0.875rem;
        border-radius: 6px;
        padding: 0.4rem 1rem;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #4F52D6;
        color: #FFFFFF;
    }

    /* Form Inputs */
    .stTextInput>div>div>input {
        background-color: #12161F;
        border: 1px solid #1F2430;
        color: #E6E8EB;
        border-radius: 6px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366F1;
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_mermaid(mermaid_code: str, height: int = 450):
    """Render a Mermaid diagram using the streamlit-mermaid package."""
    st_mermaid(mermaid_code, height=f"{height}px")


# ── Pipeline Factory Cached Per User ID ──────────────────────────────────────
@st.cache_resource(show_spinner="Initializing pipeline...")
def get_user_pipeline(user_id: str) -> RAGPipeline:
    setup_logger()
    return RAGPipeline(user_id=user_id)


# ── Session State Initialization ──────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_chunk" not in st.session_state:
    st.session_state.selected_chunk = None
if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None

# ── Authentication Gate ───────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown(
        '<div class="main-title">Document Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Multi-tenant document workspace. Sign in to access your knowledge base.</div>',
        unsafe_allow_html=True,
    )

    col_auth_left, col_auth_mid, col_auth_right = st.columns([1, 2, 1])
    with col_auth_mid:
        auth_tab_login, auth_tab_register = st.tabs(["Sign In", "Register Account"])

        with auth_tab_login:
            st.markdown("#### Sign In")
            login_username = (
                st.text_input("Username", key="login_username_input").strip().lower()
            )
            login_password = st.text_input(
                "Password", type="password", key="login_password_input"
            )

            if st.button("Sign In", use_container_width=True, key="login_submit_btn"):
                user = authenticate_user(login_username, login_password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user["username"]
                    st.session_state.user_email = user.get("email", "")
                    st.success(f"Signed in as {user['username']}")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.caption(
                "Default Accounts: `demo_user` / `demo123`, `alice` / `alice123`, `bob` / `bob123`"
            )

        with auth_tab_register:
            st.markdown("#### Create Account")
            reg_username = (
                st.text_input("Username", key="reg_username_input").strip().lower()
            )
            reg_email = st.text_input("Email", key="reg_email_input").strip()
            reg_password = st.text_input(
                "Password", type="password", key="reg_password_input"
            )

            if st.button("Register", use_container_width=True, key="reg_submit_btn"):
                if not reg_username or not reg_password:
                    st.warning("Username and password are required.")
                elif register_user(reg_username, reg_email, reg_password):
                    st.success("Account registered. You can now sign in.")
                else:
                    st.error("Username already exists or is invalid.")

    st.stop()

# ── User Logged In: Fetch User-Scoped Pipeline ────────────────────────────────
current_user_id = st.session_state.user_id
try:
    pipeline = get_user_pipeline(current_user_id)
    _pipeline_error = None
except Exception as _e:
    pipeline = None
    _pipeline_error = str(_e)


def get_ingested_docs_summary(p):
    """Group chunks by document source for current user."""
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
    st.markdown(
        f"<div style='font-size: 0.85rem; font-weight: 600; color: #E6E8EB;'>User: {current_user_id}</div>",
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", use_container_width=True, key="logout_btn"):
        st.session_state.clear()
        st.rerun()

    st.divider()

    st.markdown(
        "<div style='font-size: 0.8rem; font-weight: 600; color: #8B92A3; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;'>Document Upload</div>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Ingest Files", use_container_width=True):
            user_upload_dir = Path(f"./tmp_uploads/{current_user_id}")
            user_upload_dir.mkdir(exist_ok=True, parents=True)

            total_chunks = 0
            for uf in uploaded_files:
                tmp_path = user_upload_dir / uf.name
                tmp_path.write_bytes(uf.getvalue())

                with st.spinner(f"Processing {uf.name}..."):
                    try:
                        n = pipeline.ingest(str(tmp_path))
                        total_chunks += n
                        st.success(f"{uf.name}: {n} chunks")
                    except Exception as e:
                        st.error(f"{uf.name}: {e}")

            st.info(f"Indexed {total_chunks} total chunks.")

    st.divider()

    # Knowledge Base Stats & File Management (User-Scoped)
    st.markdown(
        "<div style='font-size: 0.8rem; font-weight: 600; color: #8B92A3; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;'>Knowledge Base Stats</div>",
        unsafe_allow_html=True,
    )
    if pipeline:
        stats = pipeline.get_stats()
        doc_summary = get_ingested_docs_summary(pipeline)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats.get("total_chunks_in_vector_store", 0)}</div><div class="stat-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )
        with col_s2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{len(doc_summary)}</div><div class="stat-label">Documents</div></div>',
                unsafe_allow_html=True,
            )

        # Ingested Files List with Delete Action
        if doc_summary:
            st.markdown(
                "<div style='font-size: 0.75rem; font-weight: 600; color: #8B92A3; margin: 0.75rem 0 0.25rem 0; text-transform: uppercase; letter-spacing: 0.05em;'>Ingested Documents</div>",
                unsafe_allow_html=True,
            )
            for d in doc_summary:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(
                        f"<div style='font-size: 0.85rem; color: #E6E8EB; font-weight: 500;'>{d['source']}</div>"
                        f"<div style='font-size: 0.75rem; color: #8B92A3;'>{d['chunk_count']} chunks</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button(
                        "Delete",
                        key=f"del_doc_{d['source']}",
                        help=f"Delete {d['source']}",
                    ):
                        pipeline.delete_document(d["source"])
                        st.success(f"Deleted {d['source']}")
                        st.rerun()

        # Collapsible Database Inspector & Reset
        with st.expander("Database Maintenance", expanded=False):
            all_chunks = pipeline.get_all_chunks()
            if not all_chunks:
                st.caption("No chunks currently in database.")
            else:
                st.write(f"Stored Chunks: {len(all_chunks)}")
                for idx, c in enumerate(all_chunks, 1):
                    st.markdown(f"**[{idx}] Source:** `{c.source}` (Page {c.page})")
                    st.caption(f"{c.text[:180]}{'...' if len(c.text) > 180 else ''}")
                    if st.button(
                        f"Delete Chunk #{idx}",
                        key=f"del_btn_{c.chunk_id}_{idx}",
                        use_container_width=True,
                    ):
                        pipeline.delete_chunk(c.chunk_id)
                        st.success(f"Deleted Chunk #{idx}")
                        st.rerun()
                    st.markdown("---")

                if st.button(
                    "Reset Knowledge Base",
                    key="clear_db_btn",
                    use_container_width=True,
                ):
                    pipeline.reset_database()
                    st.success("Database reset complete.")
                    st.rerun()
    else:
        st.error(f"Pipeline error: {_pipeline_error}")

    st.divider()

    # Layout & Feature Toggles
    st.markdown(
        "<div style='font-size: 0.8rem; font-weight: 600; color: #8B92A3; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;'>Settings & Features</div>",
        unsafe_allow_html=True,
    )
    split_view = st.checkbox("Split View Inspector", value=True)
    debug_mode = st.checkbox("Debug Retrieval Scores", value=False)
    use_hyde = st.checkbox("HyDE Retrieval", value=False)
    use_multi_query = st.checkbox("Multi-Query Expansion", value=False)

    if debug_mode:
        import logging

        logging.getLogger("rag").setLevel(logging.DEBUG)


# ── Main Layout (Split View or Single Column) ─────────────────────────────────
if split_view:
    col_chat, col_inspector = st.columns([3, 2])
else:
    col_chat = st.container()
    col_inspector = None

with col_chat:
    st.markdown(
        '<div class="main-title">Document Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-title">Workspace: <b>{current_user_id}</b></div>',
        unsafe_allow_html=True,
    )

    # ── Architecture Comparison Expander ─────────────────────────────────────
    with st.expander(
        "Why RAG vs. Pasting Documents into ChatGPT/Claude",
        expanded=False,
    ):
        st.markdown(
            "Pasting entire documents into a raw LLM prompt creates two fundamental failure modes: "
            "**context overflow** and **lack of verifiability**.\n\n"
            "A dedicated multi-tenant RAG pipeline indexes unstructured documents into isolated vector collections (`user_{user_id}`), "
            "retrieving only highest-relevance evidence chunks with verifiable citations."
        )
        st.markdown("""
| Dimension | Pasting Docs into ChatGPT / Claude | Production RAG Pipeline |
|---|---|---|
| **Document Size Limits** | Restricted by model context window; large collections overflow. | Unlimited document corpus scaled across isolated ChromaDB vector collections. |
| **Source Citations** | None or vague references; cannot verify exact page source. | Enforced `[N]` citations per claim with page numbers, text excerpts & visual previews. |
| **Hallucination Control** | High risk; LLMs improvise when facts are missing from prompt. | Low temperature ($0.1$) + strict prompt guards + automated fallback response. |
| **Data Isolation** | Multi-user chats risk prompt leakage if context windows are shared. | Per-user ChromaDB collection (`user_{id}`) & isolated BM25 index prevents leaks. |
| **Answer Relevance** | Entire document dumped as noise; subject to lost-in-the-middle degradation. | Cross-Encoder reranking filters out noise, feeding top evidence chunks to LLM. |
        """)

    # Show pipeline error if init failed
    if pipeline is None:
        st.error(f"Pipeline failed to initialize: {_pipeline_error}")
        st.stop()

    doc_count = len(get_ingested_docs_summary(pipeline))

    # Empty State Representation
    if doc_count == 0 and len(st.session_state.messages) == 0:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-title">No documents uploaded yet</div>
                Upload PDF, TXT, or Markdown documents in the sidebar to build your private knowledge base and start querying.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Chat Messages Display ────────────────────────────────────────────────
    for msg_idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]

        # User Message Card
        if role == "user":
            st.markdown(
                f'<div class="user-card"><div class="role-label">User</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        # Assistant Message Card
        else:
            with st.container():
                st.markdown(
                    f'<div class="assistant-card"><div class="role-label">Assistant</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

                # Render Citations & Interactive Chunk Selectors
                if msg.get("citations"):
                    chunks = msg.get("chunks", [])
                    st.markdown(
                        "<div style='font-size: 0.8rem; font-weight: 600; color: #8B92A3; margin: 0.75rem 0 0.25rem 0; text-transform: uppercase; letter-spacing: 0.05em;'>Citations & Sources</div>",
                        unsafe_allow_html=True,
                    )

                    cols = st.columns(min(len(msg["citations"]), 4))
                    for idx, cit in enumerate(msg["citations"]):
                        c_col = cols[idx % len(cols)]
                        rc = chunks[idx] if idx < len(chunks) else None
                        source_name = rc.chunk.source if rc else f"Source {idx+1}"
                        page_num = rc.chunk.page if rc else 1

                        with c_col:
                            if st.button(
                                f"[{idx+1}] {source_name[:14]}... p.{page_num}",
                                key=f"cit_btn_{msg_idx}_{idx}",
                                help=f"Inspect Page {page_num} of {source_name}",
                            ):
                                st.session_state.selected_chunk = rc
                                st.session_state.selected_citation = cit
                                st.rerun()

                    # Expandable Page Snapshot Previews
                    with st.expander("View Page Snapshots & Excerpts", expanded=False):
                        for idx, cit in enumerate(msg["citations"]):
                            rc = chunks[idx] if idx < len(chunks) else None
                            if rc and hasattr(rc, "chunk"):
                                src = rc.chunk.source
                                pg = rc.chunk.page
                                user_pdf_path = (
                                    Path(f"./tmp_uploads/{current_user_id}") / src
                                )
                                st.markdown(f"**[{idx+1}] `{src}` — Page {pg}**")
                                if user_pdf_path.exists() and src.lower().endswith(
                                    ".pdf"
                                ):
                                    img_bytes = get_pdf_page_image(
                                        str(user_pdf_path), pg
                                    )
                                    if img_bytes:
                                        st.image(
                                            img_bytes,
                                            caption=f"Page Snapshot — {src} (Page {pg})",
                                            use_container_width=True,
                                        )
                                st.info(f"**Text Excerpt:**\n> {rc.chunk.text}")
                                st.markdown("---")

                # Debug Metrics & Retrieval Scores
                if debug_mode and msg.get("chunks"):
                    with st.expander(
                        "Retrieval Scores & Metrics",
                        expanded=False,
                    ):
                        if "faithfulness" in msg:
                            st.markdown(
                                f'<span class="score-chip">Faithfulness: {msg["faithfulness"]:.2f}</span>'
                                f'<span class="score-chip">Relevance: {msg.get("relevance", 0.0):.2f}</span>',
                                unsafe_allow_html=True,
                            )
                        for i, rc in enumerate(msg["chunks"], 1):
                            st.markdown(
                                f'<div class="score-chip">Rerank Score: {rc.score:.4f}</div>'
                                f"<b>[{i}]</b> <i>{rc.chunk.source}</i> (Page {rc.chunk.page})<br>"
                                f"<code>{rc.chunk.text[:220]}...</code>",
                                unsafe_allow_html=True,
                            )

    # ── Query Input ───────────────────────────────────────────────────────────
    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})

        if not pipeline or pipeline.get_stats()["total_chunks_in_vector_store"] == 0:
            warning = "No documents ingested in your workspace yet. Please upload documents in the sidebar first."
            st.session_state.messages.append({"role": "assistant", "content": warning})
            st.rerun()

        elif pipeline.is_diagram_request(query):
            with st.spinner("Generating diagram from context..."):
                diag = pipeline.generate_diagram(query)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Generated **{diag.diagram_type}** diagram:",
                        "is_diagram": True,
                        "mermaid_code": diag.mermaid_code,
                        "diagram_type": diag.diagram_type,
                    }
                )
                st.rerun()

        else:
            with st.spinner("Retrieving evidence & generating answer..."):
                start = time.perf_counter()
                response = pipeline.query(
                    query, use_hyde=use_hyde, use_multi_query=use_multi_query
                )
                elapsed = time.perf_counter() - start

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
            '<div class="inspector-header">Source Inspector</div>',
            unsafe_allow_html=True,
        )

        selected_rc = st.session_state.selected_chunk

        if selected_rc and hasattr(selected_rc, "chunk"):
            chunk = selected_rc.chunk
            score = getattr(selected_rc, "score", 0.0)

            st.markdown(
                f'<div class="inspector-box">'
                f'<div style="font-size: 1rem; font-weight: 600; color: #E6E8EB; margin-bottom: 0.5rem;">{chunk.source}</div>'
                f'<div class="score-chip">Page {chunk.page}</div>'
                f'<div class="score-chip">Rerank Score: {score:.4f}</div>'
                f'<div class="score-chip">ID: {chunk.chunk_id[:12]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='font-size: 0.8rem; font-weight: 600; color: #8B92A3; margin: 1rem 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.05em;'>Text Chunk Content</div>",
                unsafe_allow_html=True,
            )
            st.info(chunk.text)

            # High-Resolution PDF Page Snapshot Preview
            user_pdf_path = Path(f"./tmp_uploads/{current_user_id}") / chunk.source
            if user_pdf_path.exists() and chunk.source.lower().endswith(".pdf"):
                st.markdown(
                    f"<div style='font-size: 0.8rem; font-weight: 600; color: #8B92A3; margin: 1rem 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.05em;'>PDF Page Snapshot (Page {chunk.page})</div>",
                    unsafe_allow_html=True,
                )
                img_bytes = get_pdf_page_image(str(user_pdf_path), chunk.page)
                if img_bytes:
                    st.image(
                        img_bytes,
                        caption=f"{chunk.source} — Page {chunk.page}",
                        use_container_width=True,
                    )
            elif not user_pdf_path.exists():
                st.caption("Upload source document to render visual page snapshots.")

        else:
            st.markdown(
                """
                <div class="empty-state-box" style="padding: 1.5rem 1rem;">
                    <div class="empty-state-title" style="font-size: 0.95rem;">No citation selected</div>
                    Click any citation badge [1], [2] to inspect chunk text and page snapshots here.
                </div>
                """,
                unsafe_allow_html=True,
            )
