"""
Document Loader - First step in the RAG pipeline.

This module loads raw text from files in the data directory.
Think of it as "opening the books" before we can study them.

Supported formats:
- .txt - Plain text files
- .pdf - PDF documents (text extraction)

Each loaded document includes metadata:
- doc_name: Original filename
- page: Page number (for PDFs) or 0 (for text files)
"""

from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader

from app.config import settings


@dataclass
class Document:
    """
    Represents a loaded document or page.

    Attributes:
        content: The raw text content
        metadata: Dict with doc_name, page, etc.
    """

    content: str
    metadata: dict


def load_text_file(file_path: Path) -> list[Document]:
    """
    Load a plain text file.

    Args:
        file_path: Path to the .txt file

    Returns:
        List with single Document (text files are one "page")
    """
    content = file_path.read_text(encoding="utf-8")
    return [
        Document(
            content=content,
            metadata={
                "doc_name": file_path.name,
                "page": 0,  # Text files don't have pages
            },
        )
    ]


def load_pdf_file(file_path: Path) -> list[Document]:
    """
    Load a PDF file, extracting text from each page.

    PDFs are loaded page-by-page because:
    1. Preserves page number for citations
    2. Handles large PDFs without memory issues
    3. Each page becomes a separate searchable unit

    Args:
        file_path: Path to the .pdf file

    Returns:
        List of Documents, one per page
    """
    documents = []
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # Skip empty pages
            documents.append(
                Document(
                    content=text,
                    metadata={
                        "doc_name": file_path.name,
                        "page": page_num + 1,  # 1-indexed for humans
                    },
                )
            )

    return documents


def load_documents(data_dir: str | None = None) -> list[Document]:
    """
    Load all documents from the data directory.

    This is the main entry point. It:
    1. Scans the data directory for supported files
    2. Loads each file with the appropriate loader
    3. Returns all documents with metadata

    Args:
        data_dir: Override data directory (uses config default if None)

    Returns:
        List of all loaded Documents

    Raises:
        FileNotFoundError: If data directory doesn't exist
    """
    data_path = Path(data_dir or settings.data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents = []

    # Map file extensions to loader functions
    loaders = {
        ".txt": load_text_file,
        ".pdf": load_pdf_file,
    }

    # Find all supported files
    for ext, loader in loaders.items():
        for file_path in data_path.glob(f"*{ext}"):
            print(f"Loading: {file_path.name}")
            docs = loader(file_path)
            documents.extend(docs)
            print(f"  → Loaded {len(docs)} page(s)")

    print(f"\nTotal: {len(documents)} documents loaded")
    return documents
