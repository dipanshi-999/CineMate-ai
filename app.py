import html
import shutil
import tempfile
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma-db"
UPLOAD_DIR = "uploaded_books"
BOOK_NAME_FILE = Path(CHROMA_DIR) / "book_name.txt"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookMind — RAG Reader",
    page_icon="📖",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0f1117; }

section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e2740;
}
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: #e8edfc;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.app-sub { color: #5a6585; font-size: 0.85rem; margin-top: 2px; margin-bottom: 1.5rem; }

.chat-row { display: flex; gap: 12px; margin-bottom: 16px; align-items: flex-start; }
.chat-row.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0; margin-top: 2px;
}
.avatar.ai  { background: #1e3a5f; }
.avatar.usr { background: #2a1f4f; }

.bubble {
    max-width: 72%; padding: 12px 16px; border-radius: 14px;
    font-size: 0.92rem; line-height: 1.6; color: #d8e0f0;
    white-space: pre-wrap; word-break: break-word;
}
.bubble.ai  { background: #161f35; border: 1px solid #1e2b4a; border-top-left-radius: 4px; }
.bubble.usr { background: #1a1040; border: 1px solid #2d1f6e; border-top-right-radius: 4px; margin-left: auto; }

.sources-wrap { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.source-pill {
    background: #0d1a30; border: 1px solid #1e3050;
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.75rem; color: #6b8fc4;
}

.empty-state { text-align: center; color: #3a4260; padding: 60px 20px; }
.empty-icon { font-size: 3.5rem; margin-bottom: 12px; }
.empty-text { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #4a5580; }
.empty-hint { font-size: 0.82rem; color: #2e3550; margin-top: 6px; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.3px; }
.badge-ready { background: #0d2e1a; color: #3dd68c; border: 1px solid #1a5c36; }
.badge-none  { background: #1a1020; color: #7855c0; border: 1px solid #3d2080; }

.section-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #3a4565; margin: 20px 0 8px 0; }

.stTextInput > div > div > input {
    background: #131825 !important; border: 1px solid #1e2b45 !important;
    color: #d8e0f0 !important; border-radius: 10px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3d5fcc !important;
    box-shadow: 0 0 0 2px rgba(61, 95, 204, 0.15) !important;
}

.stButton > button {
    background: #1e2d5a !important; color: #8aabf0 !important;
    border: 1px solid #2a3f80 !important; border-radius: 8px !important;
    font-weight: 500 !important; transition: all 0.15s;
}
.stButton > button:hover {
    background: #263775 !important; border-color: #3d5fcc !important; color: #c5d8ff !important;
}

[data-testid="stFileUploader"] {
    background: #131825 !important; border: 1px dashed #1e2b45 !important;
    border-radius: 10px !important; padding: 12px !important;
}

.stSpinner > div { border-top-color: #3d5fcc !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore_from_pdf(pdf_path: str, embedding_model) -> Chroma:
    """Load PDF → split → embed → persist to CHROMA_DIR atomically."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # FIX: build into a temp dir first; only replace CHROMA_DIR on success
    with tempfile.TemporaryDirectory() as tmp_dir:
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=tmp_dir,
        )
        if Path(CHROMA_DIR).exists():
            shutil.rmtree(CHROMA_DIR)
        shutil.copytree(tmp_dir, CHROMA_DIR)

    # Return a handle pointing at the final location
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)


def load_existing_vectorstore(embedding_model) -> tuple[Chroma | None, str | None]:
    """Return (vectorstore, book_name) if a persisted DB exists, else (None, None)."""
    if not Path(CHROMA_DIR).exists():
        return None, None
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    # FIX: read the sidecar file to recover the original book name
    name = None
    if BOOK_NAME_FILE.exists():
        name = BOOK_NAME_FILE.read_text().strip() or "Previously indexed book"
    else:
        name = "Previously indexed book"
    return vs, name


# FIX: removed @st.cache_resource so a rotated API key takes effect immediately
def load_llm():
    return init_chat_model(LLM_MODEL, model_provider="groq")


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document." """,
    ),
    (
        "human",
        """Context: {context}

Question: {question}""",
    ),
])


def ask(query: str, retriever, llm) -> tuple[str, list]:
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    sources = []
    for d in docs:
        page_num = d.metadata.get("page")
        # FIX: consistent label whether page is known or not
        label = f"p.{page_num + 1}" if page_num is not None else "p.?"
        sources.append(label)

    return response.content, list(dict.fromkeys(sources))


def reset_db():
    # FIX: also remove uploaded files on full reset
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
    if Path(UPLOAD_DIR).exists():
        shutil.rmtree(UPLOAD_DIR)
    st.session_state.vectorstore = None
    st.session_state.book_name = None
    st.session_state.messages = []


# ── Session state init ─────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vectorstore": None,
    "book_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">📖 BookMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Ask anything about your book</div>', unsafe_allow_html=True)

    if st.session_state.book_name:
        # FIX: escape book name before injecting into HTML
        safe_name = html.escape(st.session_state.book_name)
        st.markdown(
            f'<span class="badge badge-ready">✓ Ready</span>&nbsp; '
            f'<span style="font-size:0.8rem;color:#5a7a9a;">{safe_name}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge badge-none">No book loaded</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload a book</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if st.button("Process book", use_container_width=True):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            save_path = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            try:
                with st.spinner("Reading and indexing…"):
                    emb = load_embedding_model()
                    vs = build_vectorstore_from_pdf(save_path, emb)
                    # FIX: persist the book name alongside the DB
                    BOOK_NAME_FILE.write_text(uploaded.name)
                    st.session_state.vectorstore = vs
                    st.session_state.book_name = uploaded.name
                    st.session_state.messages = []
                st.success("Book indexed! Start asking questions.")
            except Exception as e:
                st.error(f"Failed to index the book: {e}")
            finally:
                # FIX: always clean up the uploaded file from disk
                try:
                    os.remove(save_path)
                except OSError:
                    pass

            st.rerun()

    # Load existing DB if nothing in session but DB exists
    if st.session_state.vectorstore is None and Path(CHROMA_DIR).exists():
        emb = load_embedding_model()
        vs, name = load_existing_vectorstore(emb)
        if vs:
            st.session_state.vectorstore = vs
            st.session_state.book_name = name

    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Remove book & reset", use_container_width=True):
        reset_db()
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.72rem;color:#2e3a55;">Powered by LangChain · Groq · ChromaDB</span>',
        unsafe_allow_html=True,
    )


# ── Main area ──────────────────────────────────────────────────────────────────
col_pad_l, col_main, col_pad_r = st.columns([0.5, 9, 0.5])

with col_main:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📚</div>
            <div class="empty-text">Upload a book and ask anything</div>
            <div class="empty-hint">BookMind reads only from your document — no hallucinations.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            if role == "user":
                # FIX: escape user content before injecting into HTML
                safe_content = html.escape(msg["content"])
                st.markdown(f"""
                <div class="chat-row user">
                    <div class="avatar usr">🧑</div>
                    <div class="bubble usr">{safe_content}</div>
                </div>""", unsafe_allow_html=True)
            else:
                # FIX: escape LLM response before injecting into HTML
                safe_content = html.escape(msg["content"])
                sources_html = ""
                if msg.get("sources"):
                    # FIX: use lstrip("p.") instead of [2:] for robustness
                    pills = "".join(
                        f'<span class="source-pill">Page {html.escape(s.lstrip("p."))}</span>'
                        for s in msg["sources"]
                    )
                    sources_html = f'<div class="sources-wrap">📌 {pills}</div>'
                st.markdown(f"""
                <div class="chat-row ai">
                    <div class="avatar ai">🤖</div>
                    <div class="bubble ai">{safe_content}{sources_html}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    query = st.chat_input(
        "Ask a question about your book…",
        disabled=st.session_state.vectorstore is None,
    )

    if query:
        if st.session_state.vectorstore is None:
            st.warning("Please upload and process a PDF book first.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})

            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
            )

            # FIX: wrap ask() in try/except to handle Groq outages gracefully
            try:
                with st.spinner("Thinking…"):
                    llm = load_llm()
                    answer, sources = ask(query, retriever, llm)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Could not get a response from the LLM: {e}")
                # Remove the user message so the turn can be retried cleanly
                st.session_state.messages.pop()

            st.rerun()