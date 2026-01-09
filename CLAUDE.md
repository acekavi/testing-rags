# RAG Q&A API Project

## Project Overview
Production-style RAG (Retrieval-Augmented Generation) Q&A API built with Python, FastAPI, and LangChain. **Runs 100% locally with Ollama - no API keys needed!**

## Tech Stack
- **Framework**: FastAPI
- **LLM**: Ollama (local, free) - Mistral, Llama, Phi, etc.
- **Vector DB**: ChromaDB (Docker)
- **Embeddings**: Sentence Transformers (local, free)
- **Language**: Python 3.11+

## Quick Start

```bash
# 1. Start ChromaDB and Ollama containers
docker compose up -d

# 2. Pull a model into Ollama (first time only, ~4GB for mistral)
docker exec -it rag-ollama ollama pull mistral

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file (no API keys needed!)
cp .env.example .env

# 6. Run the API (port 8080 to avoid conflict with ChromaDB)
uvicorn app.main:app --reload --port 8080

# 7. Open API docs
# http://localhost:8080/docs
```

## Project Structure
```
rag-excersice/
├── docker-compose.yml      # ChromaDB + Ollama containers
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
| `OLLAMA_HOST` | Ollama server host | localhost |
| `OLLAMA_PORT` | Ollama server port | 11434 |
| `LLM_MODEL_NAME` | Ollama model to use | mistral |
| `EMBEDDING_MODEL_NAME` | Local embedding model | all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Characters per chunk | 512 |
| `CHUNK_OVERLAP` | Overlap between chunks | 50 |
| `TOP_K` | Chunks to retrieve | 5 |
| `CHROMA_HOST` | ChromaDB host | localhost |
| `CHROMA_PORT` | ChromaDB port | 8000 |

## Recommended Models for CPU (No GPU)

| Model | Size | RAM | Speed | Quality |
|-------|------|-----|-------|---------|
| `gemma2:2b` | 2B | ~2GB | Fastest | Good |
| `llama3.2` | 3B | ~2GB | Fast | Good |
| `phi3` | 3.8B | ~3GB | Fast | Good |
| `mistral` | 7B | ~4GB | Medium | Best |

To switch models:
```bash
# Pull the model
docker exec -it rag-ollama ollama pull llama3.2

# Update .env
LLM_MODEL_NAME=llama3.2
```

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
6. **100% Local**: No API keys, no cloud dependencies, data stays private
