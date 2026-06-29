"""
tests/test_retriever.py - Unit tests for the Retriever agent.

Tests:
  - SJR CSV loading and ranking logic
  - Paper parsing from raw S2 API response
  - Candidate ranking (SJR over citation count)
  - DOI validation via CrossRef (mocked)
  - Rate limiting logic
  - CrossRef fallback and batch DOI resolution
  - fetch_paper_by_id and fetch_paper_by_doi
  - get_references with CrossRef-first and S2 fallback

Implementation plan coverage: Week 1 (Retriever tests)
"""

import time
from unittest.mock import patch, MagicMock, PropertyMock

import httpx
import pytest

from agents.retriever import RetrieverAgent
from core.state import CandidatePaper


class TestRetrieverInit:
    """Tests for Retriever initialisation."""

    def test_loads_sjr_index(self, base_config):
        """SJR CSV is loaded into the in-memory index."""
        agent = RetrieverAgent(base_config)
        # The SJR index should have entries (the CSV is ~30k journals)
        assert len(agent.sjr_index) > 1000

    def test_missing_sjr_csv_graceful(self, base_config):
        """Missing SJR CSV does not crash - falls back to citation count."""
        base_config["sjr_csv_path"] = "/nonexistent/path.csv"
        agent = RetrieverAgent(base_config)
        assert agent.sjr_index == {}

    def test_max_candidates_from_config(self, base_config):
        """max_candidates is read from config."""
        base_config["max_candidates"] = 5
        agent = RetrieverAgent(base_config)
        assert agent.max_candidates == 5


class TestParsePaper:
    """Tests for _parse_paper method."""

    def test_parses_full_response(self, base_config, sample_s2_paper_response):
        """Parses all fields from a complete S2 API response."""
        agent = RetrieverAgent(base_config)
        paper = agent._parse_paper(sample_s2_paper_response)
        assert isinstance(paper, CandidatePaper)
        assert paper.paper_id == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        assert paper.title == "Attention Is All You Need"
        assert paper.year == 2017
        assert paper.citation_count == 120000
        assert paper.open_access_url == "https://arxiv.org/pdf/1706.03762.pdf"
        assert paper.doi == "10.48550/arXiv.1706.03762"
        assert len(paper.authors) == 3

    def test_handles_missing_optional_fields(self, base_config):
        """Gracefully handles missing optional fields in S2 response."""
        agent = RetrieverAgent(base_config)
        minimal = {
            "paperId": "abc",
            "title": "Minimal Paper",
            "authors": [],
        }
        paper = agent._parse_paper(minimal)
        assert paper.paper_id == "abc"
        assert paper.year is None
        assert paper.open_access_url is None
        assert paper.doi is None
        assert paper.citation_count == 0

    def test_handles_null_open_access_pdf(self, base_config):
        """openAccessPdf can be null (closed-access paper)."""
        agent = RetrieverAgent(base_config)
        raw = {
            "paperId": "xyz",
            "title": "Closed Paper",
            "authors": [{"name": "Author"}],
            "year": 2020,
            "citationCount": 500,
            "openAccessPdf": None,
            "externalIds": {"DOI": "10.1000/test"},
        }
        paper = agent._parse_paper(raw)
        assert paper.open_access_url is None
        assert paper.doi == "10.1000/test"

    def test_attaches_sjr_score_from_venue(self, base_config):
        """SJR score is attached when the venue matches the SJR index."""
        agent = RetrieverAgent(base_config)
        raw = {
            "paperId": "sjr_test",
            "title": "SJR Test Paper",
            "authors": [{"name": "Author"}],
            "year": 2023,
            "citationCount": 100,
            "openAccessPdf": None,
            "publicationVenue": {"name": "Nature Reviews Molecular Cell Biology"},
            "externalIds": {},
        }
        paper = agent._parse_paper(raw)
        assert paper.sjr_score is not None
        assert paper.sjr_score > 30  # Known to be ~35.568


class TestSJRRanking:
    """Tests for the SJR ranking logic."""

    def test_sjr_beats_citation_count(self, base_config):
        """Papers with SJR scores rank above high-citation papers without SJR."""
        agent = RetrieverAgent(base_config)

        papers = [
            CandidatePaper(
                paper_id="high_cite", title="High Cite", authors=[],
                citation_count=500, sjr_score=None,
            ),
            CandidatePaper(
                paper_id="low_cite_high_sjr", title="Low Cite High SJR", authors=[],
                citation_count=100, sjr_score=5.0,
            ),
        ]

        ranked = agent._rank_by_sjr(papers)
        assert ranked[0].paper_id == "low_cite_high_sjr"

    def test_citation_count_fallback(self, base_config):
        """Papers without SJR are ranked by citation count."""
        agent = RetrieverAgent(base_config)

        papers = [
            CandidatePaper(paper_id="low", title="Low", authors=[], citation_count=10),
            CandidatePaper(paper_id="high", title="High", authors=[], citation_count=5000),
        ]

        ranked = agent._rank_by_sjr(papers)
        assert ranked[0].paper_id == "high"


class TestRetrieverRun:
    """Tests for the full Retriever.run() method with mocked HTTP."""

    @patch.object(RetrieverAgent, "_search_semantic_scholar")
    def test_returns_candidates(self, mock_search, base_config, sample_s2_paper_response):
        """run() returns parsed and ranked CandidatePaper list."""
        mock_search.return_value = [sample_s2_paper_response]
        agent = RetrieverAgent(base_config)
        result = agent.run("attention mechanism")
        assert len(result) == 1
        assert result[0].title == "Attention Is All You Need"

    @patch.object(RetrieverAgent, "_search_semantic_scholar")
    def test_empty_search_returns_empty(self, mock_search, base_config):
        """run() returns empty list when S2 returns no results."""
        mock_search.return_value = []
        agent = RetrieverAgent(base_config)
        result = agent.run("nonexistent concept xyz123")
        assert result == []

    @patch.object(RetrieverAgent, "_search_semantic_scholar")
    def test_max_candidates_limit(self, mock_search, base_config, sample_s2_paper_response):
        """run() limits results to max_candidates."""
        base_config["max_candidates"] = 2
        # Return 5 papers
        papers = [dict(sample_s2_paper_response, paperId=f"paper_{i}") for i in range(5)]
        mock_search.return_value = papers
        agent = RetrieverAgent(base_config)
        result = agent.run("attention")
        assert len(result) <= 2


class TestDOIValidation:
    """Tests for DOI validation via CrossRef."""

    @patch("agents.retriever.httpx.Client")
    def test_valid_doi_returns_true(self, mock_client_cls, base_config):
        """validate_doi returns True for a 200 response."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = MagicMock(status_code=200)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        assert agent.validate_doi("10.48550/arXiv.1706.03762") is True

    @patch("agents.retriever.httpx.Client")
    def test_invalid_doi_returns_false(self, mock_client_cls, base_config):
        """validate_doi returns False for a 404 response."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = MagicMock(status_code=404)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        assert agent.validate_doi("10.0000/nonexistent") is False

    @patch("agents.retriever.httpx.Client")
    def test_validate_doi_connection_error_returns_false(self, mock_client_cls, base_config):
        """validate_doi returns False on connection error."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.RequestError("timeout")
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        assert agent.validate_doi("10.1234/test") is False


# ── Rate-limit retry logic ─────────────────────────────────────────────────

class TestRateLimitRetry:
    """Tests for _get() exponential backoff on 429 responses."""

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_retries_on_429_then_succeeds(self, mock_client_cls, mock_sleep, base_config):
        """_get retries on 429 and returns data when it succeeds."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp_429 = MagicMock(status_code=429)
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {"data": []}
        resp_200.raise_for_status = MagicMock()
        mock_client.get.side_effect = [resp_429, resp_429, resp_200]

        agent = RetrieverAgent(base_config)
        agent._last_request_time = 0.0
        result = agent._get("http://test.com", {})
        assert result == {"data": []}
        assert mock_client.get.call_count == 3

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_all_retries_exhausted_returns_sentinel(self, mock_client_cls, mock_sleep, base_config):
        """_get returns _S2_RATE_LIMITED when all retries exhausted."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp_429 = MagicMock(status_code=429)
        mock_client.get.return_value = resp_429

        agent = RetrieverAgent(base_config)
        agent._last_request_time = 0.0
        result = agent._get("http://test.com", {})
        assert result is agent._S2_RATE_LIMITED
        # 1 initial + 4 retries = 5
        assert mock_client.get.call_count == 5

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_http_error_returns_empty(self, mock_client_cls, mock_sleep, base_config):
        """_get returns {} on non-429 HTTP errors."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp_500 = MagicMock(status_code=500)
        resp_500.text = "Internal Server Error"
        resp_500.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=resp_500,
        )
        mock_client.get.return_value = resp_500

        agent = RetrieverAgent(base_config)
        agent._last_request_time = 0.0
        result = agent._get("http://test.com", {})
        assert result == {}

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_connection_error_returns_empty(self, mock_client_cls, mock_sleep, base_config):
        """_get returns {} on connection errors."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.get.side_effect = httpx.RequestError("timeout")

        agent = RetrieverAgent(base_config)
        agent._last_request_time = 0.0
        result = agent._get("http://test.com", {})
        assert result == {}

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.time.time")
    @patch("agents.retriever.httpx.Client")
    def test_proactive_rate_limiting(self, mock_client_cls, mock_time, mock_sleep, base_config):
        """_get waits if calls are too close together."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {"ok": True}
        resp_200.raise_for_status = MagicMock()
        mock_client.get.return_value = resp_200

        # Simulate: last request at time 100, now at 100.5 (within interval)
        mock_time.side_effect = [100.5, 100.5]  # elapsed check, then set _last_request_time

        agent = RetrieverAgent(base_config)
        agent._last_request_time = 100.0
        agent._get("http://test.com", {})
        # Should have slept to wait out the interval
        mock_sleep.assert_called()

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_api_key_sent_in_headers(self, mock_client_cls, mock_sleep, base_config):
        """_get includes x-api-key header when API key is set."""
        base_config["semantic_scholar_api_key"] = "test-key-123"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {}
        resp_200.raise_for_status = MagicMock()
        mock_client.get.return_value = resp_200

        agent = RetrieverAgent(base_config)
        agent._last_request_time = 0.0
        agent._get("http://test.com", {})
        call_kwargs = mock_client.get.call_args[1]
        assert call_kwargs["headers"]["x-api-key"] == "test-key-123"


# ── fetch_paper_by_id & fetch_paper_by_doi ──────────────────────────────────

class TestFetchPaperById:
    """Tests for fetch_paper_by_id."""

    def test_returns_candidate_paper(self, base_config, sample_s2_paper_response):
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get", return_value=sample_s2_paper_response):
            paper = agent.fetch_paper_by_id("204e3073870fae3d05bcbc2f6a8e263d9b72e776")
        assert paper is not None
        assert paper.title == "Attention Is All You Need"

    def test_returns_none_on_empty_data(self, base_config):
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get", return_value={}):
            paper = agent.fetch_paper_by_id("unknown")
        assert paper is None

    def test_returns_none_on_rate_limited(self, base_config):
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get", return_value=agent._S2_RATE_LIMITED):
            paper = agent.fetch_paper_by_id("xyz")
        assert paper is None


class TestFetchPaperByDoi:
    """Tests for fetch_paper_by_doi."""

    def test_returns_candidate_paper(self, base_config, sample_s2_paper_response):
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get", return_value=sample_s2_paper_response):
            paper = agent.fetch_paper_by_doi("10.48550/arXiv.1706.03762")
        assert paper is not None
        assert paper.title == "Attention Is All You Need"

    def test_returns_none_on_rate_limited(self, base_config):
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get", return_value=agent._S2_RATE_LIMITED):
            paper = agent.fetch_paper_by_doi("10.1234/unknown")
        assert paper is None


# ── get_references with CrossRef-first fallback ─────────────────────────────

class TestGetReferences:
    """Tests for get_references with CrossRef + S2 strategies."""

    def test_s2_references_returns_paper_ids(self, base_config):
        """S2 fallback returns paper IDs when no DOI provided."""
        agent = RetrieverAgent(base_config)
        s2_data = {
            "data": [
                {"citedPaper": {"paperId": "ref1", "title": "Paper 1", "year": 2020}},
                {"citedPaper": {"paperId": "ref2", "title": "Paper 2", "year": 2019}},
                {"citedPaper": {"paperId": None}},  # null paperId
            ]
        }
        with patch.object(agent, "_get", return_value=s2_data):
            refs = agent.get_references("seed123")
        assert refs == ["ref1", "ref2"]

    def test_s2_references_empty_data(self, base_config):
        """Returns empty list when S2 has no references."""
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get", return_value={"data": []}):
            refs = agent.get_references("seed123")
        assert refs == []

    def test_s2_rate_limited_returns_empty(self, base_config):
        """Returns empty when S2 is rate-limited."""
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get", return_value=agent._S2_RATE_LIMITED):
            refs = agent.get_references("seed123")
        assert refs == []

    def test_crossref_first_when_doi_provided(self, base_config):
        """When DOI is provided, tries CrossRef first."""
        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_get_references_crossref", return_value=["cr_ref1", "cr_ref2"]) as mock_cr:
            refs = agent.get_references("seed123", doi="10.1234/test")
        assert refs == ["cr_ref1", "cr_ref2"]
        mock_cr.assert_called_once_with("10.1234/test", 20)

    def test_s2_fallback_when_crossref_empty(self, base_config):
        """Falls back to S2 when CrossRef returns nothing."""
        agent = RetrieverAgent(base_config)
        s2_data = {
            "data": [
                {"citedPaper": {"paperId": "s2_ref1", "title": "Paper", "year": 2020}},
            ]
        }
        with patch.object(agent, "_get_references_crossref", return_value=[]):
            with patch.object(agent, "_get", return_value=s2_data):
                refs = agent.get_references("seed123", doi="10.1234/test")
        assert refs == ["s2_ref1"]

    def test_s2_all_null_paper_ids(self, base_config):
        """When S2 returns entries but all have null paperId."""
        agent = RetrieverAgent(base_config)
        s2_data = {
            "data": [
                {"citedPaper": {"paperId": None}},
                {"citedPaper": {}},
            ]
        }
        with patch.object(agent, "_get", return_value=s2_data):
            refs = agent.get_references("seed123")
        assert refs == []


# ── CrossRef reference resolution ───────────────────────────────────────────

class TestCrossRefReferences:
    """Tests for _get_references_crossref and _batch_resolve_dois."""

    @patch("agents.retriever.httpx.Client")
    def test_crossref_fetches_and_resolves_dois(self, mock_client_cls, base_config):
        """CrossRef fetches references and batch-resolves DOIs to S2 IDs."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        crossref_resp = MagicMock(status_code=200)
        crossref_resp.json.return_value = {
            "message": {
                "reference": [
                    {"DOI": "10.1000/ref1"},
                    {"DOI": "10.1000/ref2"},
                    {"key": "no-doi-ref"},  # no DOI
                ]
            }
        }
        mock_client.get.return_value = crossref_resp

        agent = RetrieverAgent(base_config)
        with patch.object(agent, "_batch_resolve_dois", return_value=["s2_id_1", "s2_id_2"]):
            result = agent._get_references_crossref("10.1234/test")
        assert result == ["s2_id_1", "s2_id_2"]

    @patch("agents.retriever.httpx.Client")
    def test_crossref_non_200_returns_empty(self, mock_client_cls, base_config):
        """CrossRef non-200 returns empty list."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        resp = MagicMock(status_code=404)
        mock_client.get.return_value = resp
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        result = agent._get_references_crossref("10.1234/bad")
        assert result == []

    @patch("agents.retriever.httpx.Client")
    def test_crossref_no_references_returns_empty(self, mock_client_cls, base_config):
        """CrossRef 200 but no references returns empty."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"message": {"reference": []}}
        mock_client.get.return_value = resp
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        result = agent._get_references_crossref("10.1234/norefs")
        assert result == []

    @patch("agents.retriever.httpx.Client")
    def test_crossref_no_doi_refs_returns_empty(self, mock_client_cls, base_config):
        """CrossRef references without DOIs returns empty."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "message": {"reference": [{"key": "ref1"}, {"key": "ref2"}]}
        }
        mock_client.get.return_value = resp
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        result = agent._get_references_crossref("10.1234/nodois")
        assert result == []

    @patch("agents.retriever.httpx.Client")
    def test_crossref_connection_error_returns_empty(self, mock_client_cls, base_config):
        """CrossRef connection error returns empty list."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.RequestError("timeout")
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        result = agent._get_references_crossref("10.1234/timeout")
        assert result == []


class TestBatchResolveDois:
    """Tests for _batch_resolve_dois."""

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_batch_resolves_dois(self, mock_client_cls, mock_sleep, base_config):
        """Batch POST resolves DOIs to S2 paper IDs."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = MagicMock(status_code=200)
        resp.json.return_value = [
            {"paperId": "s2_id_1"},
            None,  # some DOIs don't resolve
            {"paperId": "s2_id_2"},
        ]
        mock_client.post.return_value = resp

        agent = RetrieverAgent(base_config)
        result = agent._batch_resolve_dois(["DOI:10.1/a", "DOI:10.1/b", "DOI:10.1/c"], limit=10)
        assert result == ["s2_id_1", "s2_id_2"]

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_batch_non_200_returns_empty(self, mock_client_cls, mock_sleep, base_config):
        """Batch endpoint non-200 returns empty."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        resp = MagicMock(status_code=500)
        mock_client.post.return_value = resp
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        result = agent._batch_resolve_dois(["DOI:10.1/a"], limit=5)
        assert result == []

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_batch_429_retries_then_succeeds(self, mock_client_cls, mock_sleep, base_config):
        """Batch retries on 429 then returns data."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp_429 = MagicMock(status_code=429)
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = [{"paperId": "id1"}]
        mock_client.post.side_effect = [resp_429, resp_200]

        agent = RetrieverAgent(base_config)
        result = agent._batch_resolve_dois(["DOI:10.1/a"], limit=5)
        assert result == ["id1"]

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_batch_all_retries_exhausted(self, mock_client_cls, mock_sleep, base_config):
        """All batch retries exhausted returns empty."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = MagicMock(status_code=429)
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        result = agent._batch_resolve_dois(["DOI:10.1/a"], limit=5)
        assert result == []

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_batch_connection_error_returns_empty(self, mock_client_cls, mock_sleep, base_config):
        """Batch connection error returns empty."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.RequestError("timeout")
        mock_client_cls.return_value = mock_client

        agent = RetrieverAgent(base_config)
        result = agent._batch_resolve_dois(["DOI:10.1/a"], limit=5)
        assert result == []

    @patch("agents.retriever.time.sleep")
    @patch("agents.retriever.httpx.Client")
    def test_batch_respects_limit(self, mock_client_cls, mock_sleep, base_config):
        """Batch result is truncated to limit."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        resp = MagicMock(status_code=200)
        resp.json.return_value = [{"paperId": f"id_{i}"} for i in range(10)]
        mock_client.post.return_value = resp

        agent = RetrieverAgent(base_config)
        result = agent._batch_resolve_dois([f"DOI:10.1/{i}" for i in range(10)], limit=3)
        assert len(result) == 3


# ── SJR edge cases ──────────────────────────────────────────────────────────

class TestSJREdgeCases:
    """Tests for SJR CSV loading edge cases."""

    def test_european_decimal_notation(self, tmp_path, base_config):
        """SJR CSV with European comma decimals is parsed correctly."""
        csv_file = tmp_path / "sjr.csv"
        csv_file.write_text(
            "Title;SJR\n"
            "Test Journal;12,345\n"
            "Another Journal;5,678\n",
            encoding="utf-8",
        )
        base_config["sjr_csv_path"] = str(csv_file)
        agent = RetrieverAgent(base_config)
        assert agent.sjr_index["test journal"] == pytest.approx(12.345)
        assert agent.sjr_index["another journal"] == pytest.approx(5.678)

    def test_zero_sjr_excluded(self, tmp_path, base_config):
        """Journals with SJR score 0 are excluded from the index."""
        csv_file = tmp_path / "sjr.csv"
        csv_file.write_text(
            "Title;SJR\n"
            "Good Journal;5,000\n"
            "Zero Journal;0,000\n",
            encoding="utf-8",
        )
        base_config["sjr_csv_path"] = str(csv_file)
        agent = RetrieverAgent(base_config)
        assert "good journal" in agent.sjr_index
        assert "zero journal" not in agent.sjr_index

    def test_invalid_sjr_value_treated_as_zero(self, tmp_path, base_config):
        """Invalid SJR values are treated as 0 (excluded)."""
        csv_file = tmp_path / "sjr.csv"
        csv_file.write_text(
            "Title;SJR\n"
            "Invalid Journal;not-a-number\n"
            "Good Journal;1,500\n",
            encoding="utf-8",
        )
        base_config["sjr_csv_path"] = str(csv_file)
        agent = RetrieverAgent(base_config)
        assert "invalid journal" not in agent.sjr_index
        assert "good journal" in agent.sjr_index


# ── Run method edge cases ───────────────────────────────────────────────────

class TestRunEdgeCases:
    """Additional run() method edge cases."""

    @patch.object(RetrieverAgent, "_search_semantic_scholar")
    def test_no_fulltext_falls_back_to_abstract_only(self, mock_search, base_config):
        """When no candidates have full-text, falls back to abstract-only."""
        papers = [
            {
                "paperId": f"p{i}",
                "title": f"Paper {i}",
                "authors": [],
                "year": 2020,
                "citationCount": 100,
                "openAccessPdf": None,  # no OA
                "externalIds": {},      # no DOI
            }
            for i in range(5)
        ]
        mock_search.return_value = papers
        agent = RetrieverAgent(base_config)
        result = agent.run("test concept")
        # Should fall back to abstract-only candidates
        assert len(result) <= base_config["max_candidates"]
        assert len(result) > 0
