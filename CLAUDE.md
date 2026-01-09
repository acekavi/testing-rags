# RAG Q&A API Project

## Project Overview
Production-style RAG (Retrieval-Augmented Generation) Q&A API built with Python, FastAPI, and LangChain.

## Tech Stack
- **Framework**: FastAPI
- **LLM**: Claude (Anthropic)
- **Vector DB**: ChromaDB (Docker)
- **Embeddings**: Sentence Transformers (local, free)
- **Language**: Python 3.11+

## Quick Start

```bash
# 1. Start ChromaDB (Docker)
docker compose up -d

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Run the API (port 8080 to avoid conflict with ChromaDB)
uvicorn app.main:app --reload --port 8080

# 6. Open API docs
# http://localhost:8080/docs
```

## Project Structure
```
rag-excersice/
├── docker-compose.yml      # ChromaDB container
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── app/
│   ├── main.py            # FastAPI entry point
│   ├── config.py          # Settings from env vars
│   ├── routes/
│   │   ├── ingest.py      # POST /ingest
│   │   └── ask.py         # POST /ask
│   └── services/
│       ├── document_loader.py  # Load txt/pdf files
│       ├── chunker.py          # Split with overlap
│       ├── embeddings.py       # Text → vectors
│       ├── vector_store.py     # ChromaDB operations
│       └── rag_chain.py        # RAG pipeline
├── data/                  # Documents to ingest
│   └── company_policies.txt   # Sample document
└── tests/
```

## API Endpoints

### POST /ingest
Processes documents from `./data/` and stores in ChromaDB.

### POST /ask
Query the RAG system:
```json
// Request
{ "question": "What is the return policy?", "top_k": 5 }

// Response
{
  "answer": "Products can be returned within 30 days...",
  "sources": [
    {"doc": "company_policies.txt", "chunk_id": 0, "score": 0.89, "snippet": "..."}
  ]
}
```

## Configuration (Environment Variables)
| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `LLM_MODEL_NAME` | Claude model | claude-3-haiku-20240307 |
| `EMBEDDING_MODEL_NAME` | Local embedding model | all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Characters per chunk | 512 |
| `CHUNK_OVERLAP` | Overlap between chunks | 50 |
| `TOP_K` | Chunks to retrieve | 5 |
| `CHROMA_HOST` | ChromaDB host | localhost |
| `CHROMA_PORT` | ChromaDB port | 8000 |

## Testing the API

```bash
# 1. Ingest documents
curl -X POST http://localhost:8080/ingest

# 2. Ask a question
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

## Key Requirements Met
1. **Chunking**: Deterministic chunking with overlap and metadata (doc_name, page, chunk_id)
2. **RAG Chain**: Answers only from context; says "I don't know" if not found
3. **Citations**: Returns citations matching retrieved chunks (no fake sources)
4. **Error Handling**: Handles "no relevant docs" gracefully
5. **Config via env vars**: All settings configurable
