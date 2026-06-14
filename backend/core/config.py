"""
core/config.py - Configuration loading and validation.

WHY EXPLICIT CONFIG VALIDATION AT STARTUP:
Failing fast on missing API keys (rather than discovering the absence mid-pipeline
at Groq API call time) is standard operational practice for LLM pipelines. A
pipeline that crashes after 3 minutes of API calls because GROQ_API_KEY was
unset wastes both quota and time.

Pydantic BaseSettings reads from environment variables automatically, giving us:
  - Type coercion (str -> int for depth caps)
  - Validation with clear error messages
  - .env file support via python-dotenv
  - No scattered os.getenv() calls throughout the codebase
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    All runtime configuration for the concept Genealogy pipeline.

    Fields are grouped by subsystem. Defaults are set conservatively
    (shallow depth, small quota limits) so that a first run on a free-tier
    Groq account does not exhaust rate limits before the user has tuned params.

    Groq rate limits for free tier (as of 2025):
      llama-3.1-8b-instant: 30 req/min, 14,400 req/day
      llama-3.3-70b-versatile: 30 req/min, 1,000 req/day
    Source: https://console.groq.com/docs/rate-limits
    """

    # ── Groq LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")

    # Lightweight model for extraction/evaluation (reactive agents)
    # WHY 8b FOR REACTIVE AGENTS: keeps token expenditure proportional -
    # extraction and validation do not require comparative reasoning.
    groq_model_fast: str = Field(
        default="llama-3.1-8b-instant",
        env="GROQ_MODEL_FAST"
    )

    # Larger model for synthesis (deliberative, comparative reasoning required)
    groq_model_reasoning: str = Field(
        default="llama-3.3-70b-versatile",
        env="GROQ_MODEL_REASONING"
    )

    # ── Semantic Scholar API ───────────────────────────────────────────────────
    # Free tier does not require an API key, but providing one raises rate limits.
    # Source: https://www.semanticscholar.org/product/api
    semantic_scholar_api_key: str = Field(default="", env="SEMANTIC_SCHOLAR_API_KEY")
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"

    # ── Pipeline constraints ───────────────────────────────────────────────────
    max_depth: int = Field(default=3, env="MAX_DEPTH")
    max_candidates: int = Field(default=3, env="MAX_CANDIDATES")

    # ── SJR ranking ───────────────────────────────────────────────────────────
    # Path to local SJR CSV dataset (SCImago Journal & Country Rank, n.d.)
    # Downloaded from: https://www.scimagojr.com/journalrank.php
    # Kept local to avoid runtime HTTP dependency for ranking (design proposal 7)
    sjr_csv_path: str = Field(
        default="data/sjr_dataset.csv",
        env="SJR_CSV_PATH"
    )

    # ── Storage ────────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/concept_genealogy",
        env="DATABASE_URL"
    )
    output_dir: str = Field(default="output", env="OUTPUT_DIR")

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config() -> Config:
    """Load and validate config. Raises ValidationError with clear message on missing keys."""
    return Config()