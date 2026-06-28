"""
tests/test_retriever.py - Unit tests for the Retriever agent.

Tests:
  - SJR CSV loading and ranking logic
  - Paper parsing from raw S2 API response
  - Candidate ranking (SJR over citation count)
  - DOI validation via CrossRef (mocked)
  - Rate limiting logic

Implementation plan coverage: Week 1 (Retriever tests)
"""

from unittest.mock import patch, MagicMock

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
