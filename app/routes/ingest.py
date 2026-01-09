"""
Ingest Endpoint - Build the vector index from documents.

POST /ingest
- Loads all documents from ./data/
- Chunks them
- Generates embeddings
- Stores in ChromaDB

This should be called once to "teach" the system about your documents.
After ingestion, you can use /ask to query them.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.document_loader import load_documents
from app.services.chunker import create_chunks
from app.services.vector_store import add_chunks, clear_collection, get_collection_stats

router = APIRouter()


class IngestResponse(BaseModel):
    """Response from the ingest endpoint."""

    message: str
    documents_loaded: int
    chunks_created: int
    collection_stats: dict


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(clear_existing: bool = True):
    """
    Ingest documents from the data directory.

    This endpoint:
    1. Optionally clears existing documents (default: True)
    2. Loads all .txt and .pdf files from ./data/
    3. Splits them into chunks with overlap
    4. Generates embeddings for each chunk
    5. Stores everything in ChromaDB

    Query Parameters:
        clear_existing: If True (default), clears existing documents before ingesting

    Returns:
        Summary of ingestion results

    Raises:
        HTTPException 404: If data directory doesn't exist
        HTTPException 500: If ingestion fails
    """
    try:
        # Step 1: Clear existing documents if requested
        if clear_existing:
            print("\n--- Clearing existing documents ---")
            clear_collection()

        # Step 2: Load documents
        print("\n--- Loading documents ---")
        documents = load_documents()

        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found in data directory. Add .txt or .pdf files to ./data/",
            )

        # Step 3: Create chunks
        print("\n--- Chunking documents ---")
        chunks = create_chunks(documents)

        # Step 4: Add to vector store (embeddings are generated here)
        print("\n--- Adding to vector store ---")
        add_chunks(chunks)

        # Step 5: Get final stats
        stats = get_collection_stats()

        return IngestResponse(
            message="Ingestion completed successfully",
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            collection_stats=stats,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
