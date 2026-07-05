"""
agents/retriever.py - Retriever agent.

ROLE IN ARCHITECTURE:
Stateless worker node. Receives a decomposed
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
  SCImago (n.d.) SJR - SCImago Journal & Country Rank.
"""

import csv
import logging
import time
from typing import Any

import httpx

from core.config import load_config
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

    # Proactive rate-limit: minimum seconds between S2 API calls.
    # S2 free tier is 1 req/s. With API key it's 10 req/s.
    _MIN_REQUEST_INTERVAL = 1.1  # seconds (free tier safe)
    _MIN_REQUEST_INTERVAL_KEYED = 0.15  # seconds (with API key)

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_candidates = config.get("max_candidates", 3)
        self.api_key = config.get("semantic_scholar_api_key", "")
        self.sjr_index = self._load_sjr_index(config.get("sjr_csv_path", "data/sjr_dataset.csv"))
        self._last_request_time: float = 0.0
        self._progress = config.get("_progress_callback", lambda msg, level="INFO": None)

    def run(self, concept: str) -> list[CandidatePaper]:
        """
        Main entry point. Returns ranked candidate papers for a concept string.

        Returns at most max_candidates papers that have full-text availability
        (either open-access PDF or Elsevier DOI). Papers with only an abstract
        are excluded from the seed selection - full-text is required for
        high-confidence claim extraction from the Gold Standard seed.
        """
        logger.info("Retriever: searching for concept='%s'", concept)

        raw_results = self._search_semantic_scholar(concept)
        if not raw_results:
            logger.warning("Retriever: no results from Semantic Scholar for '%s'", concept)
            return []

        candidates = [self._parse_paper(p) for p in raw_results if p]
        ranked = self._rank_by_sjr(candidates)

        # Filter to papers with full-text access (OA or Elsevier DOI)
        valid = [p for p in ranked if p.open_access_url or p.doi]
        logger.info(
            "Retriever: %d/%d candidates have full-text access (OA or DOI)",
            len(valid), len(ranked)
        )

        result = valid[:self.max_candidates]
        if not result and ranked:
            # Fallback: if no full-text papers found, return best abstract-only
            # rather than failing entirely (extraction will use abstract fallback)
            logger.warning("Retriever: no full-text candidates, falling back to abstract-only")
            result = ranked[:self.max_candidates]

        logger.info("Retriever: returning %d candidates", len(result))
        return result

    # Retry policy for 429 responses.
    # WHY EXPONENTIAL BACKOFF WITH JITTER:
    # The Semantic Scholar academic API enforces 1 req/second cumulative across
    # all endpoints. A fixed sleep would still cause 429s if multiple endpoints
    # are called in quick succession (e.g. search + references in the same pipeline
    # run). Exponential backoff with jitter spreads retries and avoids thundering
    # herd against the rate limit (Groq, 2025a applies same principle).
    _RETRY_DELAYS = [2.0, 5.0, 10.0, 20.0]  # seconds: 4 retries before giving up

    # Sentinel value returned when all retries are exhausted due to 429.
    # Callers can check `result is _S2_RATE_LIMITED` to distinguish from
    # "API returned empty data" (which returns {}).
    _S2_RATE_LIMITED: dict = {"__rate_limited": True}

    def _get(self, url: str, params: dict) -> dict:
        """
        Shared HTTP GET with proactive rate limiting and 429 retry.

        Proactive delay: ensures minimum interval between successive S2 requests
        so we don't trigger 429 in the first place. This is crucial during the
        BFS tracing loop which issues fetch_paper_by_id + get_references for
        each paper in quick succession.

        Retry: exponential backoff if 429 still occurs (e.g. from prior burst).
        """
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        # Proactive rate limiting - wait if we're calling too fast
        min_interval = (
            self._MIN_REQUEST_INTERVAL_KEYED if self.api_key
            else self._MIN_REQUEST_INTERVAL
        )
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        for attempt, delay in enumerate([0] + self._RETRY_DELAYS):
            if delay:
                logger.warning(
                    "Semantic Scholar 429 - waiting %.1fs before retry %d/%d",
                    delay, attempt, len(self._RETRY_DELAYS)
                )
                self._progress(
                    f"Rate limited by Semantic Scholar - waiting {delay:.0f}s (retry {attempt}/{len(self._RETRY_DELAYS)})...",
                    "WARNING",
                )
                time.sleep(delay)

            try:
                self._last_request_time = time.time()
                with httpx.Client(timeout=30.0, verify=False) as client:
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
        return self._S2_RATE_LIMITED

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
        return data.get("data") or []

    def fetch_paper_by_id(self, paper_id: str) -> CandidatePaper | None:
        """
        Fetch full metadata for a specific paper ID (used during recursive tracing
        when the Orchestrator needs to resolve a reference ID to full metadata).
        """
        data = self._get(
            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
            params={"fields": self.PAPER_FIELDS}
        )
        if not data or data is self._S2_RATE_LIMITED:
            logger.warning("Could not fetch paper %s", paper_id)
            return None
        return self._parse_paper(data)

    def validate_doi(self, doi: str) -> bool:
        """
        Check if a DOI exists by querying CrossRef. Returns True if the DOI
        resolves to an existing work, False otherwise.
        """
        try:
            with httpx.Client(timeout=15.0, verify=False) as client:
                response = client.get(
                    f"https://api.crossref.org/works/{doi}",
                    headers={"User-Agent": "ConceptGenealogy/1.0 (academic research)"},
                )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.warning("DOI validation failed for %s: %s", doi, e)
            return False

    def fetch_paper_by_doi(self, doi: str) -> CandidatePaper | None:
        """
        Fetch paper metadata from Semantic Scholar using a DOI.
        Used in DOI input mode to resolve the provided DOI to full metadata.
        """
        data = self._get(
            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/DOI:{doi}",
            params={"fields": self.PAPER_FIELDS}
        )
        if not data or data is self._S2_RATE_LIMITED:
            logger.warning("Could not fetch paper by DOI: %s", doi)
            return None
        return self._parse_paper(data)

    def get_references(self, paper_id: str, limit: int = 20, doi: str | None = None) -> list[str]:
        """
        Fetch reference paper IDs for a given paper (used by Citation Extractor
        to populate the recursive tracing queue).

        WHY REFERENCES (NOT CITATIONS):
        The system builds an inverted provenance graph - tracing backward through
        a paper's sources, not forward to its derivatives. This matches the design
        goal of finding the genealogical origin of a concept, not its influence.

        STRATEGY (CrossRef-first):
          1. If DOI is available: try CrossRef API first (no rate limit, publisher metadata)
          2. Fall back to Semantic Scholar /references endpoint
        CrossRef has no rate limit and receives publisher-deposited reference metadata
        directly from Elsevier/Springer/Wiley. Going CrossRef-first avoids wasting
        the S2 rate-limit budget on reference lookups.
        """
        # Strategy 1: CrossRef (preferred - no rate limit, direct publisher metadata)
        logger.info("get_references called: paper_id=%s, doi=%s", paper_id, doi)
        if doi:
            logger.info("Trying CrossRef first for references (DOI: %s)", doi)
            crossref_result = self._get_references_crossref(doi, limit)
            if crossref_result:
                logger.info("CrossRef: %d references for paper %s", len(crossref_result), paper_id)
                return crossref_result
            logger.info("CrossRef returned no results for DOI %s - falling back to S2", doi)

        # Strategy 2: Semantic Scholar /references endpoint
        data = self._get(
            f"{self.SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/references",
            params={"fields": "paperId,title,year", "limit": limit}
        )

        # If S2 was rate-limited, return empty (CrossRef already tried above)
        if data is self._S2_RATE_LIMITED:
            logger.warning("S2 rate-limited for references of paper %s (CrossRef already tried)", paper_id)
            return []

        refs = data.get("data") or []
        total = len(refs)
        result = [
            r["citedPaper"]["paperId"]
            for r in refs
            if r.get("citedPaper", {}).get("paperId")
        ]
        if result:
            logger.info("S2 References: %d/%d valid IDs for paper %s", len(result), total, paper_id)
            return result

        # S2 returned 0 usable references
        if total > 0:
            logger.warning(
                "S2 references endpoint returned %d entries but all have null paperId "
                "(paper %s not fully indexed)",
                total, paper_id
            )
        else:
            logger.warning("S2 references endpoint returned empty data for paper %s", paper_id)

        return []

    def _get_references_crossref(self, doi: str, limit: int = 20) -> list[str]:
        """
        Fallback: fetch reference DOIs from CrossRef, then batch-resolve them
        to Semantic Scholar paper IDs.

        CrossRef has excellent reference coverage for publisher-deposited metadata
        (especially Elsevier, Springer, Wiley). Returns S2 paper IDs for
        references that exist in Semantic Scholar's corpus.
        """
        import httpx

        try:
            with httpx.Client(timeout=30.0, verify=False) as client:
                response = client.get(
                    f"https://api.crossref.org/works/{doi}",
                    headers={"User-Agent": "ConceptGenealogy/1.0 (academic research)"},
                )
            if response.status_code != 200:
                logger.debug("CrossRef: status %d for DOI %s", response.status_code, doi)
                return []

            message = response.json().get("message", {})
            references = message.get("reference", [])
            if not references:
                logger.debug("CrossRef: no references for DOI %s", doi)
                return []

            # Extract DOIs from CrossRef references (not all have DOIs)
            ref_dois = [
                f"DOI:{r['DOI']}" for r in references if r.get("DOI")
            ]
            logger.info(
                "CrossRef: %d/%d references have DOIs for %s",
                len(ref_dois), len(references), doi
            )

            if not ref_dois:
                return []

            # Batch-resolve DOIs to S2 paper IDs (max 500 per batch)
            ref_dois = ref_dois[:limit * 3]  # fetch extra since some won't resolve
            return self._batch_resolve_dois(ref_dois, limit)

        except httpx.RequestError as e:
            logger.warning("CrossRef: connection error for DOI %s: %s", doi, e)
            return []

    def _batch_resolve_dois(self, doi_ids: list[str], limit: int) -> list[str]:
        """
        Resolve a list of DOI identifiers to Semantic Scholar paper IDs
        using the batch paper endpoint.

        Args:
            doi_ids: List of DOI strings in S2 format ("DOI:10.xxx/yyy")
            limit: Maximum number of resolved IDs to return
        """
        import httpx

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        retry_delays = [2.0, 5.0, 10.0, 20.0]

        for attempt, delay in enumerate([0] + retry_delays):
            if delay:
                logger.warning(
                    "S2 batch endpoint rate-limited - waiting %.1fs before retry %d/%d",
                    delay, attempt, len(retry_delays)
                )
                self._progress(
                    f"Rate limited by Semantic Scholar batch - waiting {delay:.0f}s (retry {attempt}/{len(retry_delays)})...",
                    "WARNING",
                )
                time.sleep(delay)

            try:
                with httpx.Client(timeout=30.0, verify=False) as client:
                    response = client.post(
                        f"{self.SEMANTIC_SCHOLAR_BASE}/paper/batch",
                        json={"ids": doi_ids},
                        params={"fields": "paperId"},
                        headers=headers,
                    )

                if response.status_code == 429:
                    continue

                if response.status_code != 200:
                    logger.debug("S2 batch: status %d", response.status_code)
                    return []

                papers = response.json()
                result = [
                    p["paperId"] for p in papers
                    if p is not None and p.get("paperId")
                ]
                logger.info(
                    "CrossRef fallback: resolved %d/%d DOIs to S2 paper IDs",
                    len(result), len(doi_ids)
                )
                return result[:limit]

            except httpx.RequestError as e:
                logger.warning("S2 batch: connection error: %s", e)
                return []

        logger.error("S2 batch: all retries exhausted for DOI resolution")
        return []

    def _parse_paper(self, raw: dict) -> CandidatePaper:
        """Convert raw Semantic Scholar API response to typed CandidatePaper."""
        authors = [a.get("name", "") for a in raw.get("authors", [])]
        oa_pdf = raw.get("openAccessPdf") or {}
        external_ids = raw.get("externalIds") or {}
        venue = raw.get("publicationVenue") or {}
        venue_name = (venue.get("name") or "").lower().strip()
        sjr_score = self.sjr_index.get(venue_name) if venue_name else None
        return CandidatePaper(
            paper_id=raw.get("paperId", ""),
            title=raw.get("title", ""),
            authors=authors,
            year=raw.get("year"),
            citation_count=raw.get("citationCount", 0),
            open_access_url=oa_pdf.get("url"),
            abstract=raw.get("abstract"),
            doi=external_ids.get("DOI"),
            sjr_score=sjr_score,
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
                        # SJR CSV uses European decimal notation (comma as decimal separator)
                        raw_sjr = (row.get("SJR", "") or "").replace(",", ".")
                        score = float(raw_sjr) if raw_sjr else 0.0
                    except ValueError:
                        score = 0.0
                    if title and score > 0:
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

    config = load_config()
    agent = RetrieverAgent({"semantic_scholar_api_key": api_key, "max_candidates": config.max_candidates, "sjr_csv_path": config.sjr_csv_path})
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