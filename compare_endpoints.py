#!/usr/bin/env python3
"""
Comparison script to test /ask vs /ask-reranked endpoints.

This script demonstrates the differences between:
- Baseline vector search (/ask)
- Cross-encoder reranking (/ask-reranked)

Usage:
    python compare_endpoints.py
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8080"

# Test questions - try different types to see reranking effects
TEST_QUESTIONS = [
    "What is the return policy?",
    "How long do I have to return a product?",
    "What are the company's working hours?",
    "Can I get a refund?",
]


def ask_baseline(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Call the baseline /ask endpoint."""
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question, "top_k": top_k},
    )
    response.raise_for_status()
    return response.json()


def ask_reranked(
    question: str, initial_k: int = 20, final_k: int = 5
) -> Dict[str, Any]:
    """Call the /ask-reranked endpoint with cross-encoder reranking."""
    response = requests.post(
        f"{BASE_URL}/ask-reranked",
        json={"question": question, "initial_k": initial_k, "final_k": final_k},
    )
    response.raise_for_status()
    return response.json()


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def print_sources(sources: list, label: str):
    """Print source information in a formatted way."""
    print(f"\n{label} Sources (Top 3):")
    print("-" * 80)
    for i, source in enumerate(sources[:3], 1):
        print(f"{i}. Doc: {source['doc']}, Chunk: {source['chunk_id']}, Score: {source['score']:.4f}")
        print(f"   Snippet: {source['snippet'][:100]}...")
    print()


def compare_question(question: str):
    """Compare baseline vs reranked for a single question."""
    print_separator()
    print(f"QUESTION: {question}")
    print_separator()

    # Call baseline endpoint
    print("Calling /ask (baseline vector search)...")
    baseline_result = ask_baseline(question)

    # Call reranked endpoint
    print("Calling /ask-reranked (with cross-encoder)...")
    reranked_result = ask_reranked(question)

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Sources comparison
    print_sources(baseline_result["sources"], "BASELINE")
    print_sources(reranked_result["sources"], "RERANKED")

    # Check if order changed
    baseline_chunks = [s["chunk_id"] for s in baseline_result["sources"][:3]]
    reranked_chunks = [s["chunk_id"] for s in reranked_result["sources"][:3]]

    print("Top 3 Chunk IDs:")
    print(f"  Baseline: {baseline_chunks}")
    print(f"  Reranked: {reranked_chunks}")
    print(f"  Order Changed: {baseline_chunks != reranked_chunks}")

    # Reranking stats
    if "reranking_stats" in reranked_result:
        stats = reranked_result["reranking_stats"]
        print(f"\nReranking Stats:")
        print(f"  Initial candidates: {stats.get('initial_candidates', 'N/A')}")
        print(f"  Final results: {stats.get('final_results', 'N/A')}")
        print(f"  Order changed: {stats.get('reranking_changed_order', 'N/A')}")
        print(f"  Original top score: {stats.get('original_top_score', 'N/A')}")
        print(f"  Reranked top score: {stats.get('reranked_top_score', 'N/A')}")

    # Answers
    print("\n" + "-" * 80)
    print("BASELINE ANSWER:")
    print("-" * 80)
    print(baseline_result["answer"])

    print("\n" + "-" * 80)
    print("RERANKED ANSWER:")
    print("-" * 80)
    print(reranked_result["answer"])


def main():
    """Run comparison for all test questions."""
    print("\n" + "=" * 80)
    print("RAG ENDPOINT COMPARISON: /ask vs /ask-reranked")
    print("=" * 80)
    print("\nThis script compares:")
    print("  • /ask: Baseline vector similarity search")
    print("  • /ask-reranked: Two-stage retrieval with cross-encoder reranking")
    print("\nWhat to look for:")
    print("  • Different chunk orders (reranking changes ranking)")
    print("  • Score differences (cross-encoder scores vs cosine similarity)")
    print("  • Answer quality differences (reranked may be more accurate)")

    # Run comparison for each test question
    for question in TEST_QUESTIONS:
        try:
            compare_question(question)
            input("\nPress Enter to continue to next question...")
        except requests.exceptions.ConnectionError:
            print("\nERROR: Could not connect to API. Is it running?")
            print("Start the API with: uvicorn app.main:app --reload --port 8080")
            return
        except Exception as e:
            print(f"\nERROR: {e}")
            continue

    print_separator()
    print("COMPARISON COMPLETE!")
    print_separator()


if __name__ == "__main__":
    main()
