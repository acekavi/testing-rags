"""
Ask Endpoint - Query the RAG system.

POST /ask
- Takes a question and optional top_k parameter
- Retrieves relevant chunks
- Generates an answer using Claude
- Returns answer with citations

This is the main user-facing endpoint.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.rag_chain import ask as rag_ask

router = APIRouter()


class AskRequest(BaseModel):
    """Request body for the ask endpoint."""

    question: str = Field(
        ...,
        description="The question to ask",
        min_length=1,
        max_length=1000,
        examples=["What is the return policy?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of chunks to retrieve (uses config default if not specified)",
        ge=1,
        le=20,
    )


class SourceResponse(BaseModel):
    """A single source/citation in the response."""

    doc: str
    chunk_id: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    """Response from the ask endpoint."""

    answer: str
    sources: list[SourceResponse]


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an answer based on ingested documents.

    The RAG pipeline:
    1. Converts your question to an embedding
    2. Finds the most similar document chunks
    3. Sends those chunks + your question to Claude
    4. Returns Claude's answer with citations

    Request Body:
        question: The question you want answered
        top_k: How many chunks to retrieve (optional, default from config)

    Returns:
        answer: The generated answer
        sources: List of chunks that were used, with relevance scores

    Example:
        POST /ask
        {"question": "What is the return policy?", "top_k": 5}

        Response:
        {
            "answer": "The return policy allows returns within 30 days...",
            "sources": [
                {"doc": "policy.txt", "chunk_id": 3, "score": 0.89, "snippet": "..."}
            ]
        }
    """
    try:
        print(f"\n--- Processing question ---")
        print(f"Question: {request.question}")
        print(f"Top K: {request.top_k or settings.top_k}")

        # Run the RAG chain
        response = rag_ask(
            question=request.question,
            top_k=request.top_k,
        )

        return AskResponse(
            answer=response.answer,
            sources=[
                SourceResponse(
                    doc=s.doc,
                    chunk_id=s.chunk_id,
                    score=s.score,
                    snippet=s.snippet,
                )
                for s in response.sources
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")
