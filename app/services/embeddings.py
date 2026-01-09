"""
Embeddings Service - Convert text to vectors (arrays of numbers).

WHAT ARE EMBEDDINGS?
====================
Embeddings are a way to represent text as numbers that capture MEANING.

Think of it like coordinates on a "meaning map":
- "happy" might be at position [0.8, 0.2, 0.5, ...]
- "joyful" would be nearby at [0.78, 0.22, 0.48, ...]  (similar meaning!)
- "sad" would be far away at [0.1, 0.9, 0.3, ...]     (opposite meaning)

WHY EMBEDDINGS?
===============
1. Computers can't compare "meaning" of text directly
2. But they CAN compare numbers (is 0.8 close to 0.78? Yes!)
3. Similar meanings = similar numbers = found by search

HOW IT WORKS
============
A neural network (the embedding model) was trained on billions of text examples
to learn that certain words/phrases have similar meanings. When you give it text,
it outputs 384 numbers (for MiniLM) that represent the "meaning coordinates".

LOCAL VS API EMBEDDINGS
=======================
- Local (what we use): Model runs on your machine, free, private
- API (OpenAI, etc.): Sends text to cloud, costs money, faster

We use sentence-transformers' all-MiniLM-L6-v2:
- Small: ~80MB download
- Fast: ~14k sentences/second on CPU
- Good quality for most use cases
"""

from sentence_transformers import SentenceTransformer

from app.config import settings

# Global model instance (loaded once, reused)
# This is a singleton pattern - avoids reloading the model for each request
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or create the embedding model (singleton).

    The model is loaded once and cached in memory.
    First call downloads the model (~80MB) if not cached.

    Returns:
        SentenceTransformer model ready to encode text
    """
    global _model

    if _model is None:
        print(f"Loading embedding model: {settings.embedding_model_name}")
        _model = SentenceTransformer(settings.embedding_model_name)
        print(f"  → Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of texts to embeddings.

    Args:
        texts: List of strings to embed

    Returns:
        List of embeddings (each embedding is a list of floats)

    Example:
        embeddings = embed_texts(["Hello world", "Goodbye world"])
        # embeddings[0] = [0.1, 0.2, ...] (384 numbers)
        # embeddings[1] = [0.15, 0.18, ...] (384 numbers)
    """
    model = get_embedding_model()

    # encode() returns numpy arrays, convert to lists for JSON serialization
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Convert a single query to an embedding.

    This is used for the user's question when searching.
    The question embedding is compared against all chunk embeddings
    to find the most similar ones.

    Args:
        query: The search query / user question

    Returns:
        Single embedding (list of floats)

    Example:
        query_embedding = embed_query("What is the return policy?")
        # query_embedding = [0.2, 0.5, ...] (384 numbers)
    """
    model = get_embedding_model()
    embedding = model.encode(query)
    return embedding.tolist()
