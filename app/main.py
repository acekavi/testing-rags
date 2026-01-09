"""
RAG Q&A API - Main Application Entry Point.

This is a production-style RAG (Retrieval-Augmented Generation) API that:
1. Ingests documents from a directory
2. Answers questions using the ingested documents
3. Provides citations for transparency

Run with:
    uvicorn app.main:app --reload

API Docs:
    http://localhost:8080/docs (Swagger UI)
    http://localhost:8080/redoc (ReDoc)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import ingest, ask


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs on startup and shutdown.
    Good place for initialization and cleanup.
    """
    # Startup
    print("\n" + "=" * 50)
    print("RAG Q&A API Starting Up (100% Local)")
    print("=" * 50)
    print(f"Ollama: {settings.ollama_url}")
    print(f"LLM Model: {settings.llm_model_name}")
    print(f"Embedding Model: {settings.embedding_model_name}")
    print(f"Chunk Size: {settings.chunk_size}")
    print(f"Chunk Overlap: {settings.chunk_overlap}")
    print(f"Top K: {settings.top_k}")
    print(f"ChromaDB: {settings.chroma_url}")
    print(f"Data Directory: {settings.data_dir}")
    print("=" * 50)
    print("\nAPI Docs: http://localhost:8080/docs")
    print("=" * 50 + "\n")

    yield  # Application runs here

    # Shutdown
    print("\nRAG Q&A API Shutting Down")


# Create FastAPI application
app = FastAPI(
    title="RAG Q&A API",
    description="""
A production-style RAG (Retrieval-Augmented Generation) API.

**Runs 100% locally with Ollama - no API keys needed!**

## What is RAG?

RAG combines document retrieval with LLM generation:
1. **Ingest**: Upload documents to create a searchable knowledge base
2. **Ask**: Query the knowledge base and get grounded answers with citations

## Endpoints

- **POST /ingest**: Process documents and build the vector index
- **POST /ask**: Ask questions and get answers with sources

## Getting Started

1. Add documents to the `./data/` directory
2. Call `POST /ingest` to process them
3. Call `POST /ask` with your questions
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware (allows requests from any origin)
# In production, you'd restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(ask.router, tags=["Q&A"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RAG Q&A API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ingest": "POST /ingest - Ingest documents from ./data/",
            "ask": "POST /ask - Ask a question",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
