"""
Reranking Service - Cross-encoder based reranking for improved retrieval accuracy.

WHAT IS RERANKING?
==================
Reranking is a two-stage retrieval approach:

Stage 1 (Fast): Use bi-encoder (sentence-transformers) to get top_k=20 candidates
  - Bi-encoder: Encodes query and documents separately
  - Fast because embeddings are pre-computed
  - Good for initial filtering from large corpus

Stage 2 (Accurate): Use cross-encoder to rerank top 20 down to final top_k=5
  - Cross-encoder: Scores query + document together as a pair
  - Slower but more accurate (sees full context of both)
  - Only runs on small candidate set (20 items)

WHY RERANKING WORKS
===================
Bi-encoders (embeddings):
  - Encode query: "What is the return policy?" → [0.1, 0.3, ...]
  - Encode document: "Products can be returned..." → [0.2, 0.4, ...]
  - Compare: cosine_similarity(query_vec, doc_vec)
  - Problem: Never sees query and document together!

Cross-encoders (reranking):
  - Input: "What is the return policy? [SEP] Products can be returned..."
  - Output: Relevance score (0-1)
  - More accurate because it sees both query and document at once
  - Can understand query-document interactions

WHEN TO USE RERANKING
======================
Use when:
  - Precision is critical (customer support, legal, medical)
  - Queries are complex or nuanced
  - Top 1-3 results matter most (not just any relevant result)

Don't use when:
  - Speed is critical (real-time, high QPS)
  - Dataset is small (< 1000 docs)
  - Simple keyword matching is enough

TRADE-OFFS
==========
Pros:
  + 10-30% accuracy improvement (measured by hit rate, MRR)
  + Works with existing embeddings (just add reranking layer)
  + Local models available (no API costs)

Cons:
  - Adds latency (~100-300ms for 20 candidates)
  - Requires GPU for best performance (CPU is slower)
  - Cannot pre-compute scores (must run at query time)

MODELS
======
We use 'cross-encoder/ms-marco-MiniLM-L-6-v2':
  - Size: ~90MB
  - Latency: ~100-200ms for 20 candidates (CPU)
  - Accuracy: Good balance of speed vs accuracy
  - Trained on MS MARCO passage ranking dataset

Alternatives:
  - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (slower, more accurate)
  - 'cross-encoder/ms-marco-TinyBERT-L-6' (faster, less accurate)
"""

from typing import Optional
from sentence_transformers import CrossEncoder

from app.config import settings
from app.services.vector_store import SearchResult

# Global cross-encoder model (singleton)
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """
    Get or create the cross-encoder reranking model (singleton).

    The model is loaded once and reused across requests.
    First load takes ~2-3 seconds, subsequent calls are instant.

    Returns:
        CrossEncoder model instance
    """
    global _reranker

    if _reranker is None:
        print(f"Loading cross-encoder model: {settings.reranker_model_name}")
        print("  (This may take a few seconds on first load...)")
        _reranker = CrossEncoder(settings.reranker_model_name)
        print(f"  → Reranker loaded successfully!")

    return _reranker


def rerank(
    query: str, results: list[SearchResult], top_k: Optional[int] = None
) -> list[SearchResult]:
    """
    Rerank search results using a cross-encoder model.

    This is the "Option B" reranking approach:
    1. Takes initial vector search results (e.g., top 20)
    2. Scores each query-document pair with cross-encoder
    3. Re-sorts by cross-encoder scores
    4. Returns top_k results (e.g., final 5)

    Args:
        query: The user's question
        results: Initial search results from vector store
        top_k: Number of final results to return (uses config default if None)

    Returns:
        Reranked and trimmed list of SearchResult objects
    """
    if not results:
        return []

    k = top_k or settings.top_k

    # If we have fewer results than requested, just return them as-is
    if len(results) <= k:
        return results

    print(f"Reranking {len(results)} candidates to top {k}...")

    # Get the cross-encoder model
    reranker = get_reranker()

    # Prepare query-document pairs for the cross-encoder
    # Format: [(query, doc1), (query, doc2), ...]
    pairs = [(query, result.content) for result in results]

    # Score all pairs with the cross-encoder
    # Returns scores in range [-10, 10] (higher = more relevant)
    scores = reranker.predict(pairs)

    # Combine results with their new scores
    reranked_results = []
    for result, score in zip(results, scores):
        # Update the score to the cross-encoder score
        # Normalize to [0, 1] range for consistency (assuming scores are mostly in [-5, 5])
        normalized_score = max(0.0, min(1.0, (score + 5) / 10))

        # Create new SearchResult with updated score
        reranked_results.append(
            SearchResult(
                content=result.content,
                doc_name=result.doc_name,
                chunk_id=result.chunk_id,
                page=result.page,
                score=round(normalized_score, 4),
            )
        )

    # Sort by score (descending) and take top_k
    reranked_results.sort(key=lambda x: x.score, reverse=True)
    final_results = reranked_results[:k]

    print(f"  → Reranked and selected top {len(final_results)} results")

    return final_results


def compare_rankings(
    query: str, original: list[SearchResult], reranked: list[SearchResult]
) -> dict:
    """
    Compare original vector search vs reranked results.

    Useful for analysis and debugging - shows how reranking changed the order.

    Args:
        query: The search query
        original: Original vector search results
        reranked: Reranked results

    Returns:
        Dict with comparison metrics
    """
    # Track which chunk IDs appear in top-k
    original_ids = [r.chunk_id for r in original]
    reranked_ids = [r.chunk_id for r in reranked]

    # Calculate overlap
    overlap = len(set(original_ids) & set(reranked_ids))
    total = len(reranked_ids)

    # Calculate rank changes
    rank_changes = []
    for i, chunk_id in enumerate(reranked_ids):
        if chunk_id in original_ids:
            original_rank = original_ids.index(chunk_id)
            new_rank = i
            rank_changes.append(abs(new_rank - original_rank))

    avg_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0

    return {
        "query": query,
        "overlap": overlap,
        "total": total,
        "overlap_percentage": round((overlap / total * 100) if total > 0 else 0, 1),
        "avg_rank_change": round(avg_rank_change, 2),
        "original_top_3": original_ids[:3],
        "reranked_top_3": reranked_ids[:3],
    }
