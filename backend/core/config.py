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

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# __file__ is core/config.py, so .parent.parent reaches the backend root
BACKEND_ROOT = Path(__file__).resolve().parent.parent

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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Groq LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str

    # Lightweight model for extraction/evaluation (stateless worker nodes)
    # WHY 8b FOR STATELESS WORKER NODES: keeps token expenditure proportional -
    # extraction and validation do not require comparative reasoning.
    groq_model_fast: str = "llama-3.1-8b-instant"

    # Larger model for synthesis (deliberative, comparative reasoning required)
    groq_model_reasoning: str = "llama-3.3-70b-versatile"

    # ── Semantic Scholar API ───────────────────────────────────────────────────
    # Free tier does not require an API key, but providing one raises rate limits.
    # Source: https://www.semanticscholar.org/product/api
    semantic_scholar_api_key: str = ""
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"

    # ── Elsevier API (ScienceDirect / Scopus) ──────────────────────────────────
    # Institutional API key for full-text retrieval of non-OA papers.
    # Allows data mining under academic license (no storage of PDFs).
    # Source: https://dev.elsevier.com/
    elsevier_api_key: str = ""

    # ── Pipeline constraints ───────────────────────────────────────────────────
    max_depth: int = 3
    max_candidates: int = 3

    # Confidence assigned to claims extracted from abstracts rather than full text.
    # Default 0.5 reflects that abstracts contain summarised findings without the
    # supporting detail present in full papers. Configurable via env to allow
    # domain-specific tuning (e.g., raise for fields with information-dense abstracts).
    abstract_confidence: float = 0.5

    # ── SJR ranking ───────────────────────────────────────────────────────────
    # Path to local SJR CSV dataset (SCImago Journal & Country Rank, n.d.)
    # Downloaded from: https://www.scimagojr.com/journalrank.php
    # Kept local to avoid runtime HTTP dependency for ranking (design proposal 7)
    sjr_csv_path: str = str(BACKEND_ROOT / "data" / "scimagojr2025.csv")

    # ── Storage ────────────────────────────────────────────────────────────────
    database_url: str = "postgresql://postgres:postgres@localhost:5432/concept_genealogy"
    output_dir: str = "output"

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"


def load_config() -> Config:
    """Load and validate config. Raises ValidationError with clear message on missing keys."""
    return Config()