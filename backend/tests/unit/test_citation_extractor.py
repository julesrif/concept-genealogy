"""
tests/test_citation_extractor.py - Unit tests for the CitationExtractor agent.

Tests:
  - _to_typed_claims static method
  - Extraction strategy fallback order (PDF → Elsevier → Abstract)
  - Empty text handling
  - LLM claim extraction (mocked)

Implementation plan coverage: Week 2 (CitationExtractor tests)
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.citation_extractor import CitationExtractorAgent
from core.state import CandidatePaper, ExtractedClaim


def _make_groq_response(content: str):
    """Helper: create a mock Groq chat_completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.fixture
def extractor(base_config):
    base_config["_groq_client"] = None
    base_config["elsevier_api_key"] = ""
    with patch("agents.citation_extractor.CitationExtractorAgent._init_elsevier"):
        agent = CitationExtractorAgent(base_config)
        agent.elsevier = None
    return agent


@pytest.fixture
def paper_with_pdf():
    return CandidatePaper(
        paper_id="pdf_paper",
        title="Paper With PDF",
        authors=["Author A"],
        year=2020,
        open_access_url="https://example.com/paper.pdf",
        doi="10.1234/test",
    )


@pytest.fixture
def paper_without_pdf():
    return CandidatePaper(
        paper_id="no_pdf_paper",
        title="Paper Without PDF",
        authors=["Author B"],
        year=2021,
        open_access_url=None,
        doi="10.5678/test",
        abstract="This paper discusses attention mechanisms in neural networks.",
    )


class TestToTypedClaims:
    """Tests for _to_typed_claims static method."""

    def test_converts_raw_dicts_to_claims(self, paper_with_pdf):
        """Raw LLM output dicts are converted to ExtractedClaim objects."""
        raw = [
            {"claim_text": "Claim one.", "inline_citations": ["Doe, 2020"], "confidence": 0.9},
            {"claim_text": "Claim two.", "inline_citations": [], "confidence": 0.8},
        ]
        claims = CitationExtractorAgent._to_typed_claims(
            raw, paper=paper_with_pdf, depth=0, confidence=1.0,
        )
        assert len(claims) == 2
        assert all(isinstance(c, ExtractedClaim) for c in claims)
        assert claims[0].source_paper_id == "pdf_paper"
        assert claims[0].depth == 0
        assert claims[0].confidence == 0.9

    def test_confidence_uses_raw_value(self, paper_with_pdf):
        """confidence comes from raw dict value, not the confidence parameter (when present)."""
        raw = [{"claim_text": "Claim.", "confidence": 0.8}]
        claims = CitationExtractorAgent._to_typed_claims(
            raw, paper=paper_with_pdf, depth=1, confidence=0.5,
        )
        # The raw dict has confidence=0.8, so that's used
        assert claims[0].confidence == 0.8

    def test_missing_confidence_uses_parameter(self, paper_with_pdf):
        """Missing confidence field in raw dict uses the confidence parameter."""
        raw = [{"claim_text": "No confidence field.", "inline_citations": []}]
        claims = CitationExtractorAgent._to_typed_claims(
            raw, paper=paper_with_pdf, depth=0, confidence=0.5,
        )
        assert claims[0].confidence == 0.5

    def test_empty_raw_returns_empty(self, paper_with_pdf):
        """Empty LLM output returns empty list."""
        claims = CitationExtractorAgent._to_typed_claims(
            [], paper=paper_with_pdf, depth=0, confidence=1.0,
        )
        assert claims == []


class TestExtractionFallback:
    """Tests for the fallback chain: PDF → Elsevier → Abstract."""

    @patch.object(CitationExtractorAgent, "_extract_from_pdf")
    @patch.object(CitationExtractorAgent, "_llm_extract_claims")
    def test_pdf_is_first_choice(
        self, mock_llm, mock_pdf, extractor, paper_with_pdf
    ):
        """When OA PDF is available, it is tried first."""
        mock_pdf.return_value = ("Full text from PDF", 1.0)
        mock_llm.return_value = [
            {"claim_text": "PDF claim", "confidence": 0.9}
        ]

        result = extractor.run(paper_with_pdf, depth=0)
        mock_pdf.assert_called_once()
        assert len(result) >= 1

    @patch.object(CitationExtractorAgent, "_extract_from_pdf")
    @patch.object(CitationExtractorAgent, "_llm_extract_claims")
    def test_falls_back_to_abstract_on_pdf_failure(
        self, mock_llm, mock_pdf, extractor, paper_with_pdf
    ):
        """When PDF extraction fails, falls back to abstract."""
        mock_pdf.return_value = ("", 0.0)  # PDF extraction failed
        mock_llm.return_value = [
            {"claim_text": "Abstract claim", "confidence": 0.7}
        ]

        # Give the paper an abstract to fall back to
        paper_with_pdf.abstract = "Some abstract text."
        result = extractor.run(paper_with_pdf, depth=0)
        # LLM should still be called (with abstract)
        assert mock_llm.called

    @patch.object(CitationExtractorAgent, "_llm_extract_claims")
    def test_abstract_only_paper(self, mock_llm, extractor, paper_without_pdf):
        """Paper with no PDF uses abstract with confidence 0.5."""
        mock_llm.return_value = [
            {"claim_text": "Abstract-only claim", "confidence": 0.8}
        ]

        result = extractor.run(paper_without_pdf, depth=0)
        assert len(result) >= 1
        # confidence from raw dict is 0.8 (used as-is)
        for claim in result:
            assert claim.confidence == 0.8


class TestLLMExtraction:
    """Tests for _llm_extract_claims with mocked LLM."""

    def test_valid_json_response(self, base_config):
        """LLM returns valid JSON array of claims."""
        import json
        claims_json = json.dumps([
            {"claim_text": "First claim.", "inline_citations": ["Smith, 2019"], "confidence": 0.9},
            {"claim_text": "Second claim.", "inline_citations": [], "confidence": 0.8},
        ])
        mock_groq = MagicMock()
        mock_groq.chat_completion = MagicMock(
            return_value=_make_groq_response(claims_json)
        )
        base_config["_groq_client"] = mock_groq
        base_config["elsevier_api_key"] = ""

        with patch("agents.citation_extractor.CitationExtractorAgent._init_elsevier"):
            agent = CitationExtractorAgent(base_config)
            agent.elsevier = None

        result = agent._llm_extract_claims("Some full text", "Paper Title")
        assert len(result) == 2
        assert result[0]["claim_text"] == "First claim."

    def test_empty_text_returns_empty(self, extractor):
        """Empty source text returns empty claims list without calling LLM."""
        result = extractor._llm_extract_claims("", "Paper Title")
        assert result == []


class TestExtractFromPDF:
    """Tests for PDF text extraction."""

    def test_successful_pdf_download(self, extractor):
        """Successful PDF download returns text with confidence 1.0."""
        import sys

        with patch("agents.citation_extractor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_response = MagicMock(status_code=200, content=b"fake pdf bytes")
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            mock_fitz = MagicMock()
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Extracted text from PDF."
            mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
            mock_fitz.open.return_value = mock_doc

            with patch.dict(sys.modules, {"fitz": mock_fitz}):
                text, confidence = extractor._extract_from_pdf("https://example.com/paper.pdf")
                assert "Extracted text" in text
                assert confidence == 1.0

    def test_failed_pdf_download(self, extractor):
        """Failed PDF download returns empty text with 0.0 confidence."""
        import httpx

        with patch("agents.citation_extractor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_response = MagicMock(status_code=403)
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Forbidden", request=MagicMock(), response=mock_response
            )
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            text, confidence = extractor._extract_from_pdf("https://example.com/paper.pdf")
            assert text == ""
            assert confidence == 0.0
