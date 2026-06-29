"""
Application configuration loaded from environment variables.

Uses pydantic-settings so every variable is validated at startup.
If a required variable is missing the app will refuse to start.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — populated from .env or real env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Supabase ──────────────────────────────────────────
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    # ── PostgreSQL (async connection string) ──────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/postgres"
    )

    # ── LLM ───────────────────────────────────────────────
    llm_api_key: str
    llm_base_url: str = "https://api.mimo.ai/v1"
    llm_model: str = "mimo-2.5-pro"

    # ── Embeddings (local — no API key needed) ─────────────
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Redis ──────────────────────────────────────────────
    redis_url: str | None = "redis://localhost:6379/0"
    redis_cache_ttl: int = 300
    redis_enabled: bool = True

    # ── App ───────────────────────────────────────────────
    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"]
    )
    prometheus_enabled: bool = True

    # ── RAG ───────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5
    similarity_threshold: float = 0.25
    fallback_threshold: float = 0.15

    # ── Evaluation ─────────────────────────────────────────
    evaluation_enabled: bool = True
    evaluation_sample_rate: float = 0.1  # evaluate 10% of queries

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of Settings (loaded once per process)."""
    return Settings()
