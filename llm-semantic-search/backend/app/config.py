"""Typed application settings, loaded from environment / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets
    anthropic_api_key: str = ""

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    claude_model: str = "claude-opus-4-8"

    # Vector store
    chroma_dir: str = "./data/chroma"
    collection_name: str = "documents"

    # Retrieval / chunking
    top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Generation
    max_answer_tokens: int = 1500

    # Pricing (USD per million tokens) for the configured Claude model, used to
    # estimate per-query cost. Defaults match claude-opus-4-8: $5 in / $25 out.
    input_price_per_mtok: float = 5.0
    output_price_per_mtok: float = 25.0


settings = Settings()
