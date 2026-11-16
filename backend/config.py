"""
ParmanaAI - Centralized Configuration Management
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "ParmanaAI"
    APP_VERSION: str = "2.0.0"
    ENV: str = "production"
    DEBUG: bool = False

    # Server & Networking
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # LLM Settings
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1024

    # Retrieval & Search
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "data/vector_store"
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.35
    DENSE_WEIGHT: float = 1.0
    SPARSE_WEIGHT: float = 1.0
    RRF_K: int = 60
    MAX_EVIDENCE_BUDGET: int = 3

    # Verification & Models
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    NLI_BATCH_SIZE: int = 16
    MAX_CLAIMS_PER_ANSWER: int = 20

    # Safety & Privacy
    MAX_QUERY_LENGTH: int = 4000
    MIN_QUERY_LENGTH: int = 3
    ENABLE_PII_REDACTION: bool = True
    ENABLE_SAFETY_GUARDS: bool = True

    # Database & Persistence
    DATABASE_URL: str = "sqlite:///./parmana_ai.db"


settings = Settings()
