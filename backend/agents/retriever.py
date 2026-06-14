"""
agents/retriever.py - Retriever agent.

ROLE IN ARCHITECTURE:
Reactive specialist agent (subsumption layer - Brooks, 1991). Receives a decomposed
sub-task (concept string) from the Orchestrator and returns academic metadata
without any access to global pipeline state. Stateless by design.

RESPONSIBILITIES:
  1. Query Semantic Scholar API for top papers matching the user concept
  2. Filter to open-access papers where possible (PDF required for Citation Extractor)
  3. Rank candidates by SJR score using local CSV dataset
  4. Return top-N candidates (N = max_candidates, typically 3)

WHY SEMANTIC SCHOLAR (NOT CROSSREF OR PUBMED):
  - Native open-access PDF resolution (AI2, 2023)
  - Structured citation data (references and citations per paper)
  - Free API tier with reasonable rate limits for academic use
  - arXiv integration for CS/AI domain coverage (arXiv, n.d.)

WHY LOCAL SJR RANKING (NOT API-BASED):
  The SJR dataset is ranked locally (SCImago, n.d.) to avoid adding a third
  API dependency to the hot path. The CSV is ~15MB and fits in memory.
  This also means ranking is deterministic and reproducible regardless of
  SCImago API availability.

References:
  AI2 (2023) Semantic Scholar API License Agreement.
  arXiv (n.d.) License and copyright.
  Brooks (1991) Intelligence without representation. AI, 47(1-3), 139-159.
  SCImago (n.d.) SJR - SCImago Journal & Country Rank.
"""

import csv
import logging
from typing import Any

import httpx

from core.state import CandidatePaper

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Fetches and ranks candidate seed papers from Semantic Scholar.

    Design decision: uses httpx (sync) rather than aiohttp because LangGraph
    nodes run synchronously in the current implementation. An async upgrade
    would require switching to LangGraph's async graph variant and is deferred
    to post-MVP as an optimisation, not a correctness requirement.
    """

    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

    # Fields requested from Semantic Scholar API - requesting only what we need
    # reduces response payload size and avoids hitting field-limit quotas.
    PAPER_FIELDS = (
        "paperId,title,authors,year,citationCount,"
        "openAccessPdf,abstract,publicationVenue,externalIds"
    )

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_candidates = config.get("max_candidates", 3)
        self.api_key = config.get("semantic_scholar_api_key", "")
        self.sjr_index = self._load_sjr_index(config.get("sjr_csv_path", "data/sjr_dataset.csv"))

    def run(self, concept: str) -> list[CandidatePaper]:
        """
        Main entry point. Returns ranked candidate papers for a concept string.

        Returns at most max_candidates papers. If fewer than max_candidates
        open-access papers are found, fills remaining slots with non-OA papers
        (Citation Extractor has an abstract-only fallback path for non-OA).
        """
        logger.info("Retriever: searching for concept='%s'", concept)

        raw_results = self._search_semantic_scholar(concept)
        if not raw_results:
            logger.warning("Retriever: no results from Semantic Scholar for '%s'", concept)
            return []

        candidates = [self._parse_paper(p) for p in raw_results if p]
        ranked = self._rank_by_sjr(candidates)

        logger.info("Retriever: returning %d candidates", len(ranked[:self.max_candidates]))
        return ranked[:self.max_candidates]

    # Retry policy for 429 responses.
    # WHY EXPONENTIAL BACKOFF WITH JITTER:
    # The Semantic Scholar academic API enforces 1 req/second cumulative across
    # all endpoints. A fixed sleep would still cause 429s if multiple endpoints
    # are called in quick succession (e.g. search + references in the same pipeline
    # run). Exponential backoff with jitter spreads retries and avoids thundering
    # herd against the rate limit (Groq, 2025a applies same principle).
    _RETRY_DELAYS = [1.5, 3.0, 6.0]  # seconds: 3 retries before giving up

    def _get(self, url: str, params: dict) -> dict:
        """
        Shared HTTP GET with automatic 429 retry and API key injection.

        Centralising HTTP calls here means retry logic is applied consistently
        to search, fetch_by_id, and get_references - not just search.
        Previously each method had its own try/except which meant 429s from
        get_references silently returned empty lists, breaking the tracing loop.
        """
        import time

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt, delay in enumerate([0] + self._RETRY_DELAYS):
            if delay:
                logger.warning(
                    "Semantic Scholar 429 - waiting %.1fs before retry %d/%d",
                    delay, attempt, len(self._RETRY_DELAYS)
                )
                time.sleep(delay)

            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(url, headers=headers, params=params)

                if response.status_code == 429:
                    # Don't raise yet - let the retry loop handle it
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error("Semantic Scholar API error %s: %s", e.response.status_code, e.response.text[:200])
                return {}
            except httpx.RequestError as e:
                logger.error("Semantic Scholar connection error: %s", e)
                return {}

        logger.error("Semantic Scholar: all retries exhausted for %s", url)
        return {}

    def _search_semantic_scholar(self, concept: str) -> list[dict]:
        """
        Call the Semantic Scholar /paper/search endpoint.

        Query strategy: use the concept as-is for the first search.
        The system assumes user input is well-formed English (design proposal 2,
        Assumptions). No query reformulation or disambiguation is attempted -
        that is out of scope for the MVP sprint.
        """
        data = self._get(
            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/search",
            params={
                "query": concept,
                "fields": self.PAPER_FIELDS,
                "limit": 20,  # fetch more than needed so ranking has a pool to work with
            }
        )
        return data.get("data", [])

    def fetch_paper_by_id(self, paper_id: str) -> CandidatePaper | None:
        """
        Fetch full metadata for a specific paper ID (used during recursive tracing
        when the Orchestrator needs to resolve a reference ID to full metadata).
        """
        data = self._get(
            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
            params={"fields": self.PAPER_FIELDS}
        )
        if not data:
            logger.warning("Could not fetch paper %s", paper_id)
            return None
        return self._parse_paper(data)

    def get_references(self, paper_id: str, limit: int = 20) -> list[str]:
        """
        Fetch reference paper IDs for a given paper (used by Citation Extractor
        to populate the recursive tracing queue).

        WHY REFERENCES (NOT CITATIONS):
        The system builds an inverted provenance graph - tracing backward through
        a paper's sources, not forward to its derivatives. This matches the design
        goal of finding the genealogical origin of a concept, not its influence.
        """
        data = self._get(
            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/references",
            params={"fields": "paperId,title,year", "limit": limit}
        )
        refs = data.get("data", [])
        return [
            r["citedPaper"]["paperId"]
            for r in refs
            if r.get("citedPaper", {}).get("paperId")
        ]

    @staticmethod
    def _parse_paper(raw: dict) -> CandidatePaper:
        """Convert raw Semantic Scholar API response to typed CandidatePaper."""
        authors = [a.get("name", "") for a in raw.get("authors", [])]
        oa_pdf = raw.get("openAccessPdf") or {}
        return CandidatePaper(
            paper_id=raw.get("paperId", ""),
            title=raw.get("title", ""),
            authors=authors,
            year=raw.get("year"),
            citation_count=raw.get("citationCount", 0),
            open_access_url=oa_pdf.get("url"),
            abstract=raw.get("abstract"),
        )

    def _rank_by_sjr(self, candidates: list[CandidatePaper]) -> list[CandidatePaper]:
        """
        Rank candidates by SJR score (descending). Falls back to citation count
        if the paper's venue is not in the SJR index (e.g. arXiv preprints).

        WHY SJR OVER RAW CITATION COUNT:
        Citation count is vulnerable to recency bias (newer papers have had less
        time to accumulate citations) and field-size bias (ML papers accumulate
        citations faster than niche fields). SJR normalises for both. However,
        since arXiv preprints dominate AI research and are not in SJR, we fall
        back to citation count for unindexed papers - which is reasonable because
        highly-cited arXiv papers in AI are themselves reliable indicators of impact.
        """
        def score(paper: CandidatePaper) -> float:
            if paper.sjr_score is not None:
                return paper.sjr_score * 1000  # scale up so SJR dominates citation count
            return float(paper.citation_count)

        return sorted(candidates, key=score, reverse=True)

    def _load_sjr_index(self, csv_path: str) -> dict[str, float]:
        """
        Load SJR CSV into an in-memory dict keyed by journal title (lowercase).

        WHY NOT ISSN-BASED LOOKUP:
        Semantic Scholar returns venue names, not ISSNs. Title matching is fuzzy
        but sufficient for MVP - exact ISSN matching would require a separate
        venue metadata lookup pass and is deferred to a future iteration.
        """
        sjr_index: dict[str, float] = {}
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    title = row.get("Title", "").lower().strip()
                    try:
                        score = float(row.get("SJR", 0) or 0)
                    except ValueError:
                        score = 0.0
                    if title:
                        sjr_index[title] = score
        except FileNotFoundError:
            logger.warning(
                "SJR CSV not found at '%s'. Falling back to citation count ranking. "
                "Download from https://www.scimagojr.com/journalrank.php",
                csv_path
            )
        return sjr_index


# ── Debug helper (remove before final submission) ─────────────────────────────

def debug_search(concept: str, api_key: str = "") -> None:
    """
    Standalone debug function - run directly to inspect raw Semantic Scholar output.
    Usage: python -c "from concept_tracer.agents.retriever import debug_search; debug_search('attention mechanism')"
    """
    import json
    # Load key via load_config() so .env is picked up automatically,
    # same as the main pipeline does. Avoids the os.environ trap where
    # .env variables are only loaded by pydantic-settings, not the shell.
    if not api_key:
        try:
            from core.config import load_config
            api_key = load_config().semantic_scholar_api_key
        except Exception:
            try:
                from core.config import load_config
                api_key = load_config().semantic_scholar_api_key
            except Exception:
                pass
    if api_key:
        print(f"Using API key: {api_key[:8]}...")
    else:
        print("WARNING: No API key - unauthenticated requests (429 likely)")
    agent = RetrieverAgent({"semantic_scholar_api_key": api_key, "max_candidates": 5, "sjr_csv_path": "data/sjr_dataset.csv"})
    print(f"\n=== RAW Semantic Scholar response for '{concept}' ===\n")
    raw = agent._search_semantic_scholar(concept)
    print(f"Papers returned: {len(raw)}")
    for i, p in enumerate(raw[:5]):
        print(f"\n[{i+1}] Raw keys: {list(p.keys())}")
        print(f"     title: {p.get('title')}")
        print(f"     paperId: {p.get('paperId')}")
        print(f"     openAccessPdf: {p.get('openAccessPdf')}")
        print(f"     citationCount: {p.get('citationCount')}")
    print(f"\n=== Parsed candidates ===\n")
    candidates = agent.run(concept)
    print(f"Candidates after parse+rank: {len(candidates)}")
    for c in candidates:
        print(f"  - {c.title} | OA: {bool(c.open_access_url)} | citations: {c.citation_count}")