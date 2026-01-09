"""
Text Chunker - Split documents into smaller pieces.

WHY CHUNKING?
=============
Imagine you have a 100-page manual and someone asks "What's the return policy?"
You wouldn't send the entire manual to answer this - you'd find the relevant page.

Chunking does the same thing:
1. Splits documents into smaller pieces (~500 chars each)
2. Each chunk can be individually searched and retrieved
3. Only relevant chunks are sent to the LLM (not entire documents)

OVERLAP EXPLAINED
=================
Without overlap, we might cut a sentence in half:

    Chunk 1: "Our return policy allows returns within"
    Chunk 2: "30 days of purchase with original receipt."

With overlap, the sentence appears complete in at least one chunk:

    Chunk 1: "Our return policy allows returns within 30 days"
    Chunk 2: "returns within 30 days of purchase with original receipt."

The overlap ensures we don't lose context at chunk boundaries.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.document_loader import Document


@dataclass
class Chunk:
    """
    A chunk of text with its metadata.

    Attributes:
        content: The chunk text
        metadata: Original doc metadata + chunk_id
    """

    content: str
    metadata: dict


def create_chunks(documents: list[Document]) -> list[Chunk]:
    """
    Split documents into overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter which:
    1. Tries to split on paragraph breaks first
    2. Falls back to sentence breaks
    3. Falls back to word breaks
    4. This preserves natural text boundaries when possible

    Args:
        documents: List of loaded Documents

    Returns:
        List of Chunks with metadata including chunk_id

    Example:
        docs = load_documents()
        chunks = create_chunks(docs)
        # chunks[0].metadata = {'doc_name': 'policy.txt', 'page': 0, 'chunk_id': 0}
    """
    # Configure the splitter
    # RecursiveCharacterTextSplitter tries these separators in order:
    # ["\n\n", "\n", " ", ""] - paragraphs, lines, words, characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,  # Use character count
        is_separator_regex=False,
    )

    chunks = []
    chunk_id = 0  # Global chunk ID across all documents

    for doc in documents:
        # Split this document's content
        text_chunks = splitter.split_text(doc.content)

        for text in text_chunks:
            # Create chunk with combined metadata
            chunk = Chunk(
                content=text,
                metadata={
                    **doc.metadata,  # Copy original metadata (doc_name, page)
                    "chunk_id": chunk_id,
                },
            )
            chunks.append(chunk)
            chunk_id += 1

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"  Chunk size: {settings.chunk_size} chars")
    print(f"  Overlap: {settings.chunk_overlap} chars")

    return chunks
