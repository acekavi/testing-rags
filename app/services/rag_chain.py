"""
RAG Chain - The heart of the system.

This module orchestrates the full RAG pipeline:
1. RETRIEVE: Find relevant chunks from the vector store
2. AUGMENT: Build a prompt with the question + retrieved context
3. GENERATE: Ask the LLM to answer based on the context

THE RAG PROMPT
==============
The key to RAG is the prompt. We tell the LLM:
- "Here is some context"
- "Answer the question using ONLY this context"
- "If you can't find the answer, say I don't know"

This "grounds" the LLM - it can only use information we provide,
preventing hallucinations (making stuff up).

CITATIONS
=========
We track which chunks were used to answer the question.
This provides transparency - users can verify the answer
by checking the source documents.

OLLAMA
======
Ollama runs LLMs locally on your machine:
- No API keys needed
- Data stays private (never leaves your machine)
- Free to use
- Supports many models (Mistral, Llama, Phi, etc.)
"""

from dataclasses import dataclass, asdict
import ollama

from app.config import settings
from app.services.vector_store import search, SearchResult


# The RAG prompt template
# This is critical - it instructs the LLM how to behave
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Answer the question using ONLY the information in the context below
2. If the context doesn't contain enough information to answer, say "I don't know based on the available documents"
3. Be concise and direct in your answers
4. Do not make up information that isn't in the context

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class Source:
    """
    A citation source for the answer.

    Attributes:
        doc: Source document name
        chunk_id: Chunk identifier
        score: Relevance score (0-1)
        snippet: Preview of the chunk content
    """

    doc: str
    chunk_id: int
    score: float
    snippet: str


@dataclass
class RAGResponse:
    """
    Complete response from the RAG chain.

    Attributes:
        answer: The generated answer
        sources: List of sources (citations)
    """

    answer: str
    sources: list[Source]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "answer": self.answer,
            "sources": [asdict(s) for s in self.sources],
        }


def format_context(results: list[SearchResult]) -> str:
    """
    Format search results into a context string for the prompt.

    Each chunk is labeled with its source for traceability.

    Args:
        results: Search results from vector store

    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant documents found."

    context_parts = []
    for i, result in enumerate(results, 1):
        # Format: [Source: doc.txt, Page: 1] Content here...
        source_label = f"[Source: {result.doc_name}"
        if result.page > 0:
            source_label += f", Page: {result.page}"
        source_label += "]"

        context_parts.append(f"{source_label}\n{result.content}")

    return "\n\n---\n\n".join(context_parts)


def create_snippet(content: str, max_length: int = 150) -> str:
    """
    Create a short snippet from chunk content.

    Used for the "snippet" field in citations.

    Args:
        content: Full chunk content
        max_length: Maximum snippet length

    Returns:
        Truncated content with "..." if needed
    """
    if len(content) <= max_length:
        return content

    # Try to cut at a word boundary
    truncated = content[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.7:  # Don't cut too early
        truncated = truncated[:last_space]

    return truncated + "..."


def get_ollama_client() -> ollama.Client:
    """
    Create Ollama client connected to the Docker container.

    Returns:
        Ollama client instance
    """
    return ollama.Client(host=settings.ollama_url)


def ask(question: str, top_k: int | None = None) -> RAGResponse:
    """
    Run the full RAG pipeline to answer a question.

    This is the main entry point for the RAG system:
    1. Search for relevant chunks
    2. Build the prompt with context
    3. Ask Ollama to generate an answer
    4. Return answer with citations

    Args:
        question: The user's question
        top_k: Number of chunks to retrieve (uses config default if None)

    Returns:
        RAGResponse with answer and sources
    """
    k = top_k or settings.top_k

    # Step 1: RETRIEVE - Find relevant chunks
    print(f"Searching for relevant chunks (top_k={k})...")
    search_results = search(question, top_k=k)

    if not search_results:
        return RAGResponse(
            answer="I don't know - no documents have been ingested yet. Please run the /ingest endpoint first.",
            sources=[],
        )

    print(f"  → Found {len(search_results)} relevant chunks")

    # Step 2: AUGMENT - Build the prompt
    context = format_context(search_results)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # Step 3: GENERATE - Ask Ollama (local LLM)
    print(f"Generating answer with Ollama ({settings.llm_model_name})...")
    client = get_ollama_client()

    response = client.chat(
        model=settings.llm_model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the answer text
    answer = response["message"]["content"]

    # Build citations from search results
    sources = [
        Source(
            doc=result.doc_name,
            chunk_id=result.chunk_id,
            score=result.score,
            snippet=create_snippet(result.content),
        )
        for result in search_results
    ]

    print("  → Answer generated successfully")
    return RAGResponse(answer=answer, sources=sources)
