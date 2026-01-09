# RAG Q&A API Project

## Project Overview
Production-style RAG (Retrieval-Augmented Generation) Q&A API built with Python, FastAPI, and LangChain.

## Tech Stack
- **Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **Vector DB**: FAISS or Chroma (local)
- **Embeddings**: OpenAI or local models
- **Language**: Python 3.11+

## Project Structure
```
rag-excersice/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration via env vars
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── ingest.py        # POST /ingest endpoint
│   │   └── ask.py           # POST /ask endpoint
│   └── services/
│       ├── __init__.py
│       ├── document_loader.py   # Document loading (txt/pdf)
│       ├── chunker.py           # Chunking with overlap + metadata
│       ├── embeddings.py        # Embedding generation
│       ├── vector_store.py      # Vector DB operations
│       └── rag_chain.py         # RAG chain with citations
├── data/                    # Document storage
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## API Endpoints

### POST /ingest
Builds the vector index from documents in `./data/`

### POST /ask
Query the RAG system
- Request: `{ "question": "...", "top_k": 5 }`
- Response:
```json
{
  "answer": "...",
  "sources": [
    {"doc": "a.txt", "chunk_id": 2, "score": 0.81, "snippet": "..."}
  ]
}
```

## Configuration (Environment Variables)
- `LLM_MODEL_NAME` - Model to use for generation
- `EMBEDDING_MODEL_NAME` - Model for embeddings
- `CHUNK_SIZE` - Size of document chunks (default: 512)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 50)
- `TOP_K` - Default number of results to retrieve (default: 5)
- `VECTOR_DB_TYPE` - "faiss" or "chroma"

## Key Requirements
1. **Chunking**: Deterministic chunking with overlap and metadata (doc_name, page, chunk_id)
2. **RAG Chain**: Answer only from context; if not found say "I don't know"
3. **Citations**: Return citations matching retrieved chunks (no fake sources)
4. **Error Handling**: Handle "no relevant docs" gracefully

## Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app.main:app --reload

# Run tests
pytest
```

## Quality Standards
- Clean code structure with separation of concerns
- Type hints throughout
- Environment-based configuration
- Proper error handling
- No hallucinated citations
