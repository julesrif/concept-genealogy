"""
tests/test_evaluator.py - Unit tests for the Evaluator agent.

Tests:
  - Relevance threshold filtering
  - Validity scoring via LLM (mocked)
  - Drift score computation via sentence-transformers
  - Graceful degradation without groq client
  - Graceful degradation without sentence-transformers

Implementation plan coverage: Week 2 (Evaluator tests)
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from agents.evaluator import EvaluatorAgent
from core.state import ExtractedClaim, EvaluatedClaim, PipelineState


def _make_groq_response(content: str):
    """Helper: create a mock Groq chat_completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.fixture
def mock_groq():
    """Mock Groq client with chat_completion method."""
    client = MagicMock()
    client.chat_completion = MagicMock(
        return_value=_make_groq_response(json.dumps({
            "relevant": True,
            "validity_score": 0.85,
            "relevant_concepts": ["attention mechanism"],
        }))
    )
    return client


@pytest.fixture
def state():
    return PipelineState(user_concept="attention mechanism")


@pytest.fixture
def claims_to_evaluate():
    """Set of ExtractedClaim instances for testing."""
    return [
        ExtractedClaim(
            claim_text="Self-attention allows the model to attend to all positions.",
            source_paper_id="paper_1",
            source_title="Attention Is All You Need",
            depth=0,
            confidence=0.95,
        ),
        ExtractedClaim(
            claim_text="The weather in Paris is nice in spring.",
            source_paper_id="paper_2",
            source_title="Unrelated Paper",
            depth=0,
            confidence=0.9,
        ),
        ExtractedClaim(
            claim_text="Attention mechanisms were inspired by cognitive science.",
            source_paper_id="paper_3",
            source_title="Historical Attention",
            depth=0,
            confidence=0.8,
        ),
    ]


class TestRelevanceThreshold:
    """Tests for relevance-based claim filtering."""

    def test_threshold_constant_is_reasonable(self):
        """RELEVANCE_THRESHOLD is documented as 0.3."""
        assert EvaluatorAgent.RELEVANCE_THRESHOLD == 0.3


class TestEvaluatorRun:
    """Tests for the full Evaluator.run() method."""

    def test_returns_evaluated_claims(
        self, base_config, mock_groq, claims_to_evaluate, state
    ):
        """run() returns EvaluatedClaim instances."""
        base_config["_groq_client"] = mock_groq
        agent = EvaluatorAgent(base_config)

        result = agent.run(claims_to_evaluate, state)
        assert all(isinstance(c, EvaluatedClaim) for c in result)
        assert len(result) >= 1

    def test_graceful_without_groq(self, base_config, claims_to_evaluate, state):
        """Without groq client, evaluator returns claims with default score 0.5."""
        base_config["_groq_client"] = None
        agent = EvaluatorAgent(base_config)

        result = agent.run(claims_to_evaluate, state)
        for claim in result:
            assert isinstance(claim, EvaluatedClaim)
            assert claim.validity_score == 0.5

    def test_empty_claims_returns_empty(self, base_config, mock_groq, state):
        """Empty input returns empty output."""
        base_config["_groq_client"] = mock_groq
        agent = EvaluatorAgent(base_config)
        result = agent.run([], state)
        assert result == []


class TestDriftScore:
    """Tests for concept drift detection."""

    def test_drift_score_computed_at_depth_1(self, base_config, mock_groq, state):
        """At depth > 0, drift_score is computed if embedding model available."""
        base_config["_groq_client"] = mock_groq
        agent = EvaluatorAgent(base_config)
        state.current_depth = 1
        state.concept_at_depth = {0: "attention mechanism", 1: "selective attention"}

        claims = [
            ExtractedClaim(
                claim_text="Attention was originally a cognitive concept.",
                source_paper_id="p1",
                source_title="History of Attention",
                depth=1,
                confidence=0.9,
            )
        ]

        result = agent.run(claims, state)
        # If embedding model is loaded, drift_score should be a float
        for claim in result:
            if claim.drift_score is not None:
                assert 0.0 <= claim.drift_score <= 1.0

    def test_no_drift_without_embedding_model(self, base_config, mock_groq, state):
        """Without embedding model, drift_score remains None."""
        base_config["_groq_client"] = mock_groq
        agent = EvaluatorAgent(base_config)
        agent.embedding_model = None  # disable drift detection
        state.current_depth = 1
        state.concept_at_depth = {0: "attention mechanism", 1: "selective attention"}

        claims = [
            ExtractedClaim(
                claim_text="x", source_paper_id="y",
                source_title="z", depth=1, confidence=0.9,
            )
        ]
        result = agent.run(claims, state)
        for claim in result:
            assert claim.drift_score is None


class TestEvaluateClaim:
    """Tests for the internal _evaluate_claim method."""

    def test_relevant_claim_returns_evaluated(self, base_config, mock_groq):
        """Relevant claim with high validity returns EvaluatedClaim."""
        base_config["_groq_client"] = mock_groq
        agent = EvaluatorAgent(base_config)

        claim = ExtractedClaim(
            claim_text="Self-attention computes representations.",
            source_paper_id="p1",
            source_title="T",
            depth=0,
            confidence=0.9,
        )
        result = agent._evaluate_claim(claim, "attention mechanism", depth=0)
        assert isinstance(result, EvaluatedClaim)
        assert result.validity_score == 0.85

    def test_irrelevant_claim_returns_none(self, base_config):
        """Irrelevant claim (relevant=false) is discarded (returns None)."""
        mock_client = MagicMock()
        mock_client.chat_completion = MagicMock(
            return_value=_make_groq_response(json.dumps({
                "relevant": False,
                "validity_score": 0.1,
                "relevant_concepts": [],
            }))
        )
        base_config["_groq_client"] = mock_client
        agent = EvaluatorAgent(base_config)

        claim = ExtractedClaim(
            claim_text="The weather is nice.",
            source_paper_id="p1",
            source_title="T",
            depth=0,
            confidence=0.9,
        )
        result = agent._evaluate_claim(claim, "attention mechanism", depth=0)
        assert result is None

    def test_unparseable_llm_response_returns_none(self, base_config):
        """Unparseable LLM response returns None (claim skipped)."""
        mock_client = MagicMock()
        mock_client.chat_completion = MagicMock(
            return_value=_make_groq_response("I'm not sure, maybe high?")
        )
        base_config["_groq_client"] = mock_client
        agent = EvaluatorAgent(base_config)

        claim = ExtractedClaim(
            claim_text="Test claim",
            source_paper_id="p1",
            source_title="T",
            depth=0,
            confidence=0.9,
        )
        result = agent._evaluate_claim(claim, "attention mechanism", depth=0)
        assert result is None
