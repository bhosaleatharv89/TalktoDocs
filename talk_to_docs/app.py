"""Streamlit entry point for Talk to Docs RAG application."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import settings
from core.chunker import TextChunker
from core.cleaner import TextCleaner
from core.embedder import get_embedder
from core.loader import DocumentLoader, FileValidationError
from core.rag_pipeline import RAGPipeline
from core.retriever import Retriever
from core.vector_store import FaissVectorStore
from services.ingestion_service import IngestionService
from services.llm_service import get_llm_service
from utils.logger import get_logger


logger = get_logger(__name__)

st.set_page_config(page_title="Talk to Docs", page_icon="📚", layout="wide")
st.title("📚 Talk to Docs")
st.caption("Upload documents, build a vector index, and ask grounded questions.")


@st.cache_resource
def build_services() -> tuple[IngestionService, RAGPipeline, FaissVectorStore]:
    """Build long-lived components with Streamlit resource caching."""
    vector_store = FaissVectorStore()
    vector_store.load()

    embedder = get_embedder()
    loader = DocumentLoader()
    cleaner = TextCleaner()
    chunker = TextChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)

    ingestion_service = IngestionService(loader, cleaner, chunker, embedder, vector_store)
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    rag_pipeline = RAGPipeline(retriever=retriever, llm_service=get_llm_service())
    return ingestion_service, rag_pipeline, vector_store


ingestion_service, rag_pipeline, vector_store = build_services()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Upload & Index")
    uploaded_file = st.file_uploader("Supported: PDF, TXT, DOCX", type=["pdf", "txt", "docx"])
    if st.button("Ingest Document", use_container_width=True, disabled=uploaded_file is None):
        try:
            result = ingestion_service.ingest_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
            st.success(f"Indexed '{result.file_name}' into {result.chunks_indexed} chunks.")
        except FileValidationError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001
            logger.exception("Ingestion failed")
            st.error(f"Failed to ingest document: {error}")

    st.subheader("Index Status")
    total_vectors = vector_store.index.ntotal if vector_store.index else 0
    st.metric("Indexed Chunks", total_vectors)
    st.write(f"Index name: `{settings.index_name}`")
    st.write(f"Index location: `{Path(vector_store.index_path)}`")

    if st.button("Clear Index & Chat", use_container_width=True):
        vector_store.clear()
        st.session_state.chat_history = []
        st.success("Cleared index and conversation history.")

with right_col:
    st.subheader("Ask Questions")
    user_question = st.chat_input("Ask about your uploaded documents...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        try:
            answer, citations = rag_pipeline.answer(user_question)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
            )
        except Exception as error:  # noqa: BLE001
            logger.exception("RAG question failed")
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Sorry, something went wrong: {error}",
                    "citations": [],
                }
            )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("citations"):
                st.caption("Sources")
                for citation in message["citations"]:
                    st.caption(
                        f"[{citation['rank']}] {citation['source']} (similarity={citation['score']})"
                    )
