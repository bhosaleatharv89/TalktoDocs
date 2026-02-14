# Talk to Docs (Production-Ready RAG with Streamlit)

A complete Retrieval-Augmented Generation (RAG) application that lets users upload documents (PDF/TXT/DOCX), index them with vector embeddings, and ask grounded questions through a chat UI.

## Folder Structure

```text
talk_to_docs/
│
├── app.py
├── config.py
├── requirements.txt
├── .env.example
│
├── core/
│   ├── loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│
├── services/
│   ├── llm_service.py
│   ├── ingestion_service.py
│
├── utils/
│   ├── logger.py
│   ├── helpers.py
│
└── data/
    ├── uploads/
    └── indexes/
```

## Module Overview

- **`config.py`**: Centralized environment/config loading with typed settings.
- **`utils/logger.py`**: Central logger formatter/initializer.
- **`utils/helpers.py`**: File validation, text normalization, and batching helpers.
- **`core/loader.py`**: Upload validation + PDF/TXT/DOCX text extraction.
- **`core/cleaner.py`**: Text normalization and cleanup.
- **`core/chunker.py`**: Overlapping chunk generation with metadata.
- **`core/embedder.py`**: Embedding providers (SentenceTransformer/OpenAI), batching, caching.
- **`core/vector_store.py`**: FAISS wrapper with metadata persistence and reload support.
- **`core/retriever.py`**: Similarity retrieval, top-k handling, and threshold filtering.
- **`core/rag_pipeline.py`**: Prompt assembly with retrieved context and grounded answer generation.
- **`services/llm_service.py`**: LLM abstraction with OpenAI and local fallback modes.
- **`services/ingestion_service.py`**: End-to-end ingestion pipeline from file to vector store.
- **`app.py`**: Streamlit frontend with upload panel, index metrics, chat, history, and reset.

## Installation

1. Create virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

2. Install dependencies:

```bash
pip install -r talk_to_docs/requirements.txt
```

3. Configure environment:

```bash
cp talk_to_docs/.env.example .env
# then edit .env with your keys/settings
```

## Running Locally

```bash
cd talk_to_docs
streamlit run app.py
```

Open the shown local URL (usually `http://localhost:8501`).

## Example `.env`

```dotenv
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=sentence_transformer
OPENAI_API_KEY=your-openai-api-key
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
DATA_DIR=./data
UPLOAD_DIR=./data/uploads
INDEX_DIR=./data/indexes
INDEX_NAME=default_index
MAX_FILE_SIZE_MB=10
CHUNK_SIZE=700
CHUNK_OVERLAP=120
RETRIEVER_TOP_K=5
RETRIEVER_SCORE_THRESHOLD=0.30
LOG_LEVEL=INFO
```

## Performance Optimizations Included

- `st.cache_resource` for long-lived model/vector components.
- Batching for embedding generation.
- LRU caching for repeated OpenAI embedding calls.
- Normalized embeddings + inner-product FAISS retrieval for fast similarity search.
- Index persistence to disk to avoid re-indexing after restart.

## Error Handling and Reliability

- Centralized logger with module-level context.
- Validation for file type and max file size.
- User-friendly Streamlit errors for upload/RAG failures.
- Retry logic for OpenAI completion calls.
- Safe fallback LLM mode when external API is not configured.

## Sample UI Screenshot Layout (Text Description)

- **Left panel**: file uploader, “Ingest Document” button, index metrics, and “Clear Index & Chat.”
- **Right panel**: chat input and conversation history.
- **Assistant messages**: include answer text plus source chunk references with similarity scores.

## Deployment Notes

- Keep `.env` out of version control.
- Use a production process manager/container for Streamlit deployment.
- Mount a persistent volume for `talk_to_docs/data/indexes` in production.
