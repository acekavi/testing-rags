"""
Vector Store - Interface to ChromaDB for storing and searching embeddings.

WHAT IS A VECTOR DATABASE?
==========================
A vector database is optimized for "similarity search" - finding items
that are most similar to a query, based on their vector representations.

Regular database: "Find all rows where name = 'John'"  (exact match)
Vector database:  "Find the 5 most similar items to this vector" (similarity)

HOW SIMILARITY SEARCH WORKS
===========================
1. Each chunk is stored with its embedding (384 numbers)
2. When you search, your query is also converted to an embedding
3. ChromaDB compares your query embedding to ALL stored embeddings
4. Returns the chunks whose embeddings are "closest" to yours

"Closeness" is measured by cosine similarity - the angle between vectors.
Vectors pointing the same direction = similar meaning.

CHROMADB CONCEPTS
=================
- Collection: Like a table, stores documents with embeddings
- Document: The original text
- Embedding: The vector representation
- Metadata: Additional info (doc_name, page, chunk_id)
- ID: Unique identifier for each entry
"""

from dataclasses import dataclass
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.chunker import Chunk
from app.services.embeddings import embed_texts, embed_query

# Global ChromaDB client (singleton)
_client: chromadb.HttpClient | None = None


@dataclass
class SearchResult:
    """
    A single search result from the vector store.

    Attributes:
        content: The chunk text
        doc_name: Source document name
        chunk_id: Chunk identifier
        page: Page number (0 for text files)
        score: Similarity score (0-1, higher = more similar)
    """

    content: str
    doc_name: str
    chunk_id: int
    page: int
    score: float


def get_chroma_client() -> chromadb.HttpClient:
    """
    Get or create ChromaDB HTTP client (singleton).

    Connects to the ChromaDB Docker container.

    Returns:
        ChromaDB client connected to the server
    """
    global _client

    if _client is None:
        print(f"Connecting to ChromaDB at {settings.chroma_url}")
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Test connection
        heartbeat = _client.heartbeat()
        print(f"  → Connected! Heartbeat: {heartbeat}")

    return _client


def get_collection():
    """
    Get or create the document collection.

    A collection is like a table in a regular database.
    All our document chunks are stored in one collection.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"description": "RAG document chunks"},
    )
    return collection


def add_chunks(chunks: list[Chunk]) -> int:
    """
    Add chunks to the vector store.

    This is the "ingestion" step:
    1. Generate embeddings for all chunks
    2. Store chunks with their embeddings and metadata
    3. Return count of added chunks

    Args:
        chunks: List of Chunks to add

    Returns:
        Number of chunks added
    """
    if not chunks:
        return 0

    collection = get_collection()

    # Extract texts and metadata
    texts = [chunk.content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{chunk.metadata['chunk_id']}" for chunk in chunks]

    # Generate embeddings
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_texts(texts)

    # Add to ChromaDB
    print("Adding to vector store...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"  → Added {len(chunks)} chunks to collection '{settings.chroma_collection_name}'")
    return len(chunks)


def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    """
    Search for chunks similar to the query.

    This is the "retrieval" step:
    1. Convert query to embedding
    2. Find top_k most similar chunks
    3. Return chunks with similarity scores

    Args:
        query: The search query / user question
        top_k: Number of results (uses config default if None)

    Returns:
        List of SearchResult, sorted by similarity (highest first)
    """
    k = top_k or settings.top_k
    collection = get_collection()

    # Check if collection is empty
    if collection.count() == 0:
        print("Warning: Collection is empty. Run ingestion first.")
        return []

    # Embed the query
    query_embedding = embed_query(query)

    # Search ChromaDB
    # include=["documents", "metadatas", "distances"] returns everything we need
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Convert to SearchResult objects
    search_results = []

    # Results come as lists of lists (for batch queries), we only have 1 query
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    for doc, meta, distance in zip(documents, metadatas, distances):
        # ChromaDB returns distance (lower = better)
        # Convert to similarity score (higher = better) for intuition
        # Cosine distance is in [0, 2], similarity = 1 - (distance / 2)
        similarity = 1 - (distance / 2)

        search_results.append(
            SearchResult(
                content=doc,
                doc_name=meta.get("doc_name", "unknown"),
                chunk_id=meta.get("chunk_id", -1),
                page=meta.get("page", 0),
                score=round(similarity, 4),
            )
        )

    return search_results


def clear_collection() -> None:
    """
    Delete all documents from the collection.

    Useful for re-ingesting documents from scratch.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
        print(f"Deleted collection '{settings.chroma_collection_name}'")
    except Exception:
        print(f"Collection '{settings.chroma_collection_name}' doesn't exist")


def get_collection_stats() -> dict:
    """
    Get statistics about the current collection.

    Returns:
        Dict with count and collection name
    """
    collection = get_collection()
    return {
        "collection_name": settings.chroma_collection_name,
        "document_count": collection.count(),
    }
