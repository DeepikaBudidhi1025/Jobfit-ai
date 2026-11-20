"""Central configuration, loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible endpoint works: Groq (free tier), OpenAI, Ollama, etc.
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Local embedding model (runs on CPU via fastembed, no API key needed)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Retrieval / matching
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    skill_match_threshold: float = float(os.getenv("SKILL_MATCH_THRESHOLD", "0.62"))


settings = Settings()
