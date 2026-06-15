"""
Shared pytest fixtures for the Concept Genealogy test suite.

Provides mock configurations, sample data objects, and fixtures that mirror
real Semantic Scholar / Groq responses without making network calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Configuration fixtures ──────────────────────────────────────────────────

@pytest.fixture
def base_config():
    """Minimal config dict used by all agents."""
    return {
        "semantic_scholar_api_key": "",
        "elsevier_api_key": "",
        "groq_api_key": "test-key",
        "groq_model_fast": "llama-3.1-8b-instant",
        "groq_model_reasoning": "llama-3.3-70b-versatile",
        "max_candidates": 3,
        "max_depth": 3,
        "papers_per_level": 3,
        "sjr_csv_path": str(Path(__file__).resolve().parent.parent / "data" / "scimagojr2025.csv"),
    }


@pytest.fixture
def mock_groq_client():
    """Mock Groq client that returns controllable responses."""
    client = MagicMock()

    def make_response(content: str):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        return resp

    client.chat_completion = MagicMock(
        return_value=make_response('[]')
    )
    client._make_response = make_response
    return client


# ── Sample data fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_s2_paper_response():
    """Raw Semantic Scholar API response for a single paper."""
    return {
        "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        "title": "Attention Is All You Need",
        "authors": [
            {"name": "Ashish Vaswani"},
            {"name": "Noam Shazeer"},
            {"name": "Niki Parmar"},
        ],
        "year": 2017,
        "citationCount": 120000,
        "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762.pdf"},
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "publicationVenue": {"name": "Neural Information Processing Systems"},
        "externalIds": {"DOI": "10.48550/arXiv.1706.03762"},
    }


@pytest.fixture
def sample_candidate_paper():
    """A typed CandidatePaper for use in tests."""
    from core.state import CandidatePaper
    return CandidatePaper(
        paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        citation_count=120000,
        sjr_score=None,
        open_access_url="https://arxiv.org/pdf/1706.03762.pdf",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        doi="10.48550/arXiv.1706.03762",
    )


@pytest.fixture
def sample_extracted_claims():
    """List of ExtractedClaim objects for testing Evaluator/Inspector."""
    from core.state import ExtractedClaim
    return [
        ExtractedClaim(
            claim_text="The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions.",
            source_paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            source_title="Attention Is All You Need",
            depth=0,
            inline_citations=["Bahdanau et al., 2014"],
            confidence=1.0,
        ),
        ExtractedClaim(
            claim_text="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            source_paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            source_title="Attention Is All You Need",
            depth=0,
            inline_citations=[],
            confidence=1.0,
        ),
        ExtractedClaim(
            claim_text="Self-attention computes a representation of a sequence by relating different positions of the sequence to one another.",
            source_paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            source_title="Attention Is All You Need",
            depth=0,
            inline_citations=["Lin et al., 2017"],
            confidence=0.9,
        ),
    ]


@pytest.fixture
def sample_evaluated_claims():
    """List of EvaluatedClaim objects for testing Inspector/Synthesiser."""
    from core.state import EvaluatedClaim
    return [
        EvaluatedClaim(
            claim_text="The Transformer architecture relies entirely on attention mechanisms.",
            source_paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            source_title="Attention Is All You Need",
            depth=0,
            validity_score=0.95,
            relevant_concepts=["attention mechanism", "transformer"],
            drift_score=None,
        ),
        EvaluatedClaim(
            claim_text="Multi-head attention allows joint attention to different subspaces.",
            source_paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            source_title="Attention Is All You Need",
            depth=0,
            validity_score=0.88,
            relevant_concepts=["multi-head attention"],
            drift_score=None,
        ),
    ]


@pytest.fixture
def sample_pipeline_state():
    """A mid-pipeline PipelineState for testing agent interactions."""
    from core.state import PipelineState, PipelineStage
    return PipelineState(
        stage=PipelineStage.TRACING,
        user_concept="attention mechanism",
        current_depth=0,
        max_depth=3,
        papers_per_level=3,
        concept_at_depth={0: "attention mechanism"},
    )


@pytest.fixture
def golden_fixture_claims_json():
    """
    Golden fixture: known-good LLM output for claim extraction from
    'Attention Is All You Need' abstract. Used to verify Inspector parsing
    and Citation Extractor output pipeline without needing a live LLM call.
    """
    import json
    return json.dumps([
        {
            "claim_text": "The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "inline_citations": ["Bahdanau et al., 2014", "Gehring et al., 2017"],
            "confidence": 0.95
        },
        {
            "claim_text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
            "inline_citations": [],
            "confidence": 0.9
        },
        {
            "claim_text": "The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over existing best results by over 2 BLEU.",
            "inline_citations": ["Wu et al., 2016"],
            "confidence": 0.85
        },
        {
            "claim_text": "Self-attention relates different positions of a single sequence to compute a representation of the sequence.",
            "inline_citations": ["Lin et al., 2017", "Parikh et al., 2016"],
            "confidence": 0.88
        },
        {
            "claim_text": "The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.",
            "inline_citations": [],
            "confidence": 0.8
        }
    ])
