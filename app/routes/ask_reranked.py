"""
Ask Reranked Endpoint - Query the RAG system with cross-encoder reranking.

POST /ask-reranked
- Takes a question and optional parameters
- Retrieves MORE candidates initially (e.g., top_k=20)
- Reranks with cross-encoder to get final top_k (e.g., 5)
- Generates an answer using Ollama
- Returns answer with citations

This endpoint implements "Option B: Reranking" from the RAG retrieval strategies.

COMPARISON WITH /ask:
=====================
/ask (baseline):
  - Retrieves top_k=5 chunks using vector similarity
  - Directly uses those 5 chunks for generation
  - Fast: ~50-100ms retrieval

/ask-reranked (this endpoint):
  - Retrieves top_k=20 candidates using vector similarity
  - Reranks candidates with cross-encoder to get best 5
  - Uses reranked top 5 for generation
  - Slower but more accurate: ~150-300ms retrieval + reranking

WHEN TO USE RERANKED VS BASELINE:
==================================
Use /ask-reranked when:
  - Precision is critical (you need the BEST answer)
  - Complex or nuanced queries
  - Multi-hop reasoning required

Use /ask when:
  - Speed is more important than perfect precision
  - Simple factual queries
  - High QPS scenarios
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.vector_store import search
from app.services.reranker import rerank
from app.services.rag_chain import (
    format_context,
    get_ollama_client,
    create_snippet,
    Source,
    RAG_PROMPT_TEMPLATE,
)

router = APIRouter()


class AskRerankedRequest(BaseModel):
    """Request body for the ask-reranked endpoint."""

    question: str = Field(
        ...,
        description="The question to ask",
        min_length=1,
        max_length=1000,
        examples=["What is the return policy?"],
    )
    initial_k: Optional[int] = Field(
        default=None,
        description="Number of initial candidates to retrieve before reranking (default: 20)",
        ge=5,
        le=50,
    )
    final_k: Optional[int] = Field(
        default=None,
        description="Number of final chunks after reranking (uses config default if not specified)",
        ge=1,
        le=20,
    )


class SourceResponse(BaseModel):
    """A single source/citation in the response."""

    doc: str
    chunk_id: int
    score: float  # This will be the cross-encoder score after reranking
    snippet: str


class AskRerankedResponse(BaseModel):
    """Response from the ask-reranked endpoint."""

    answer: str
    sources: list[SourceResponse]
    reranking_stats: dict  # Stats about the reranking process


@router.post("/ask-reranked", response_model=AskRerankedResponse)
async def ask_question_reranked(request: AskRerankedRequest):
    """
    Ask a question with cross-encoder reranking for improved accuracy.

    This endpoint uses a two-stage retrieval approach:
    1. Stage 1: Retrieve initial_k candidates (default 20) using vector similarity
    2. Stage 2: Rerank with cross-encoder to get final_k best results (default 5)
    3. Generate answer using the reranked chunks

    Request Body:
        question: The question you want answered
        initial_k: Number of candidates to retrieve before reranking (default: 20)
        final_k: Number of final chunks after reranking (default from config: 5)

    Returns:
        answer: The generated answer
        sources: List of reranked chunks with cross-encoder scores
        reranking_stats: Stats showing the reranking process

    Example:
        POST /ask-reranked
        {
            "question": "What is the return policy?",
            "initial_k": 20,
            "final_k": 5
        }

        Response:
        {
            "answer": "The return policy allows returns within 30 days...",
            "sources": [
                {"doc": "policy.txt", "chunk_id": 3, "score": 0.92, "snippet": "..."}
            ],
            "reranking_stats": {
                "initial_candidates": 20,
                "final_results": 5,
                "reranking_changed_order": true
            }
        }
    """
    try:
        # Default values
        initial_k = request.initial_k or settings.reranker_initial_k
        final_k = request.final_k or settings.top_k

        print(f"\n--- Processing question (with reranking) ---")
        print(f"Question: {request.question}")
        print(f"Initial K (candidates): {initial_k}")
        print(f"Final K (after reranking): {final_k}")

        # Stage 1: Retrieve initial candidates using vector similarity
        print(f"Stage 1: Retrieving {initial_k} candidates with vector search...")
        initial_results = search(request.question, top_k=initial_k)

        if not initial_results:
            return AskRerankedResponse(
                answer="I don't know - no documents have been ingested yet. Please run the /ingest endpoint first.",
                sources=[],
                reranking_stats={
                    "initial_candidates": 0,
                    "final_results": 0,
                    "reranking_changed_order": False,
                },
            )

        print(f"  → Retrieved {len(initial_results)} candidates")

        # Stage 2: Rerank with cross-encoder
        print(f"Stage 2: Reranking to top {final_k}...")
        reranked_results = rerank(
            query=request.question, results=initial_results, top_k=final_k
        )

        # Check if reranking changed the order
        original_top_ids = [r.chunk_id for r in initial_results[:final_k]]
        reranked_top_ids = [r.chunk_id for r in reranked_results]
        order_changed = original_top_ids != reranked_top_ids

        print(f"  → Reranked to {len(reranked_results)} results")
        print(f"  → Order changed: {order_changed}")

        # Stage 3: Generate answer using reranked chunks
        print(f"Stage 3: Generating answer with Ollama ({settings.llm_model_name})...")
        context = format_context(reranked_results)
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context, question=request.question
        )

        client = get_ollama_client()
        response = client.chat(
            model=settings.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response["message"]["content"]
        print("  → Answer generated successfully")

        # Build sources with cross-encoder scores
        sources = [
            Source(
                doc=result.doc_name,
                chunk_id=result.chunk_id,
                score=result.score,  # Cross-encoder score
                snippet=create_snippet(result.content),
            )
            for result in reranked_results
        ]

        # Calculate score improvement (compare top result scores)
        original_top_score = initial_results[0].score if initial_results else 0
        reranked_top_score = reranked_results[0].score if reranked_results else 0

        return AskRerankedResponse(
            answer=answer,
            sources=[
                SourceResponse(
                    doc=s.doc,
                    chunk_id=s.chunk_id,
                    score=s.score,
                    snippet=s.snippet,
                )
                for s in sources
            ],
            reranking_stats={
                "initial_candidates": len(initial_results),
                "final_results": len(reranked_results),
                "reranking_changed_order": order_changed,
                "original_top_score": round(original_top_score, 4),
                "reranked_top_score": round(reranked_top_score, 4),
                "original_top_3_chunks": original_top_ids[:3],
                "reranked_top_3_chunks": reranked_top_ids,
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}"
        )
