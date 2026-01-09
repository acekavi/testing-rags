"""
Configuration management using Pydantic Settings.

Pydantic Settings automatically:
1. Loads values from environment variables
2. Validates types (e.g., CHUNK_SIZE must be an int)
3. Provides defaults when env vars are missing
4. Raises clear errors for missing required values (like API keys)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Usage:
        from app.config import settings
        print(settings.chunk_size)  # 512
    """

    # ----- LLM Configuration -----
    anthropic_api_key: str  # Required - no default, will error if missing
    llm_model_name: str = "claude-3-haiku-20240307"

    # ----- Embedding Configuration -----
    # This model runs locally on your machine
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ----- Chunking Configuration -----
    chunk_size: int = 512      # Characters per chunk
    chunk_overlap: int = 50    # Overlap to preserve context

    # ----- Retrieval Configuration -----
    top_k: int = 5  # Number of chunks to retrieve

    # ----- ChromaDB Configuration -----
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "documents"

    # ----- Data Configuration -----
    data_dir: str = "./data"

    @property
    def chroma_url(self) -> str:
        """Full URL for ChromaDB HTTP client."""
        return f"http://{self.chroma_host}:{self.chroma_port}"

    # Tell Pydantic to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # CHUNK_SIZE and chunk_size both work
    )


# Create a singleton instance
# All modules import this same instance
settings = Settings()
