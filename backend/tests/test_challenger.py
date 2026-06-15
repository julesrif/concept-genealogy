"""
tests/test_challenger.py - Tests for the Challenger adversarial review agent.

Tests the multi-turn debate mechanics, evidence summarisation, fallback
behaviour, and output formatting.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.challenger import ChallengerAgent, MAX_DEBATE_ROUNDS
from core.state import TracingResult, EvaluatedClaim


@pytest.fixture
def mock_groq():
    """Mock Groq client returning structured challenger JSON."""
    client = MagicMock()
    client.chat_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "issues": [
                {
                    "type": "unsupported",
                    "severity": "high",
                    "quote": "The transformer revolutionised all of NLP",
                    "explanation": "No traced evidence supports this universal claim"
                },
                {
                    "type": "overstated",
                    "severity": "medium",
                    "quote": "definitively proved",
                    "explanation": "Validity score is only 0.6 — not definitive"
                }
            ],
            "overall_assessment": "adequate",
            "recommendation": "revise"
        })))]
    )
    return client


@pytest.fixture
def sample_tracing_results():
    """Tracing results with claims at multiple depths."""
    return [
        TracingResult(
            paper_id="paper1",
            paper_title="Attention Is All You Need",
            depth=0,
            doi="10.48550/arXiv.1706.03762",
            claims=[
                EvaluatedClaim(
                    claim_text="Self-attention replaces recurrence",
                    source_paper_id="paper1",
                    source_title="Attention Is All You Need",
                    depth=0,
                    validity_score=0.9,
                    relevant_concepts=["self-attention", "transformer"],
                ),
                EvaluatedClaim(
                    claim_text="Multi-head attention allows parallel computation",
                    source_paper_id="paper1",
                    source_title="Attention Is All You Need",
                    depth=0,
                    validity_score=0.85,
                    relevant_concepts=["multi-head attention"],
                ),
            ],
        ),
        TracingResult(
            paper_id="paper2",
            paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
            depth=1,
            claims=[
                EvaluatedClaim(
                    claim_text="Masked language modelling enables bidirectional context",
                    source_paper_id="paper2",
                    source_title="BERT",
                    depth=1,
                    validity_score=0.75,
                    relevant_concepts=["masked LM"],
                    drift_score=0.3,
                ),
            ],
        ),
    ]


class TestChallengerInit:
    """Tests for Challenger initialisation."""

    def test_init_with_groq(self, mock_groq):
        config = {"_groq_client": mock_groq, "groq_model_slow": "llama-3.3-70b-versatile"}
        agent = ChallengerAgent(config)
        assert agent.groq is mock_groq
        assert agent.model == "llama-3.3-70b-versatile"

    def test_init_without_groq(self):
        config = {"_groq_client": None}
        agent = ChallengerAgent(config)
        assert agent.groq is None

    def test_default_model(self, mock_groq):
        config = {"_groq_client": mock_groq}
        agent = ChallengerAgent(config)
        assert agent.model == "llama-3.3-70b-versatile"


class TestChallengerRun:
    """Tests for the main run() method."""

    def test_no_synthesis_returns_message(self, mock_groq, sample_tracing_results):
        config = {"_groq_client": mock_groq}
        agent = ChallengerAgent(config)
        result = agent.run(None, sample_tracing_results)
        assert "no synthesis to review" in result

    def test_empty_synthesis_returns_message(self, mock_groq, sample_tracing_results):
        config = {"_groq_client": mock_groq}
        agent = ChallengerAgent(config)
        result = agent.run("", sample_tracing_results)
        assert "no synthesis to review" in result

    def test_produces_critique_with_issues(self, mock_groq, sample_tracing_results):
        config = {"_groq_client": mock_groq}
        agent = ChallengerAgent(config)
        result = agent.run(
            "The transformer revolutionised all of NLP and definitively proved attention is optimal.",
            sample_tracing_results,
            user_concept="attention mechanism",
        )
        assert "Adversarial Review" in result
        assert "Unsupported claim" in result
        assert "Overstated conclusion" in result
        assert "revise" in result.lower() or "REVISE" in result

    def test_includes_coverage_summary(self, mock_groq, sample_tracing_results):
        config = {"_groq_client": mock_groq}
        agent = ChallengerAgent(config)
        result = agent.run(
            "Some synthesis text.",
            sample_tracing_results,
            user_concept="attention",
        )
        assert "Claims evaluated: 3" in result
        assert "Papers traced: 2" in result

    def test_early_termination_on_strong_assessment(self, sample_tracing_results):
        """If first round finds no issues, debate terminates early."""
        mock_groq = MagicMock()
        mock_groq.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "issues": [],
                "overall_assessment": "strong",
                "recommendation": "approve"
            })))]
        )
        config = {"_groq_client": mock_groq}
        agent = ChallengerAgent(config)
        result = agent.run("Good synthesis.", sample_tracing_results)
        # Should only call LLM once (early termination)
        assert mock_groq.chat_completion.call_count == 1
        assert "No significant issues" in result


class TestChallengerFallback:
    """Tests for fallback behaviour without LLM."""

    def test_fallback_without_groq(self, sample_tracing_results):
        config = {"_groq_client": None}
        agent = ChallengerAgent(config)
        result = agent.run(
            "Some synthesis text.",
            sample_tracing_results,
            user_concept="attention",
        )
        assert "no LLM available" in result
        assert "Claims reviewed: 3" in result
        assert "Papers traced: 2" in result

    def test_fallback_reports_high_drift_claims(self):
        """Fallback mode reports claims with high drift scores."""
        results = [
            TracingResult(
                paper_id="p1", paper_title="Paper 1", depth=1,
                claims=[
                    EvaluatedClaim(
                        claim_text="Drifted claim",
                        source_paper_id="p1", source_title="Paper 1",
                        depth=1, validity_score=0.8, drift_score=0.85,
                    ),
                ],
            )
        ]
        config = {"_groq_client": None}
        agent = ChallengerAgent(config)
        result = agent.run("Synthesis.", results)
        assert "High-drift claims" in result
        assert "1" in result  # should report 1 high-drift claim


class TestChallengerLLMFailure:
    """Tests for graceful degradation on LLM errors."""

    def test_llm_exception_returns_empty_issues(self, sample_tracing_results):
        """If LLM call raises, the round returns no issues gracefully."""
        mock_groq = MagicMock()
        mock_groq.chat_completion.side_effect = RuntimeError("API timeout")
        config = {"_groq_client": mock_groq}
        agent = ChallengerAgent(config)
        result = agent.run("Synthesis.", sample_tracing_results)
        # Should still produce a formatted critique (just with no issues)
        assert "Adversarial Review" in result
        assert "No significant issues" in result


class TestEvidenceSummary:
    """Tests for evidence summary building."""

    def test_builds_summary_with_paper_info(self, sample_tracing_results):
        config = {"_groq_client": None}
        agent = ChallengerAgent(config)
        summary = agent._build_evidence_summary(sample_tracing_results)
        assert "Attention Is All You Need" in summary
        assert "depth=0" in summary
        assert "validity=" in summary
        assert "Self-attention replaces recurrence" in summary

    def test_includes_drift_scores(self, sample_tracing_results):
        config = {"_groq_client": None}
        agent = ChallengerAgent(config)
        summary = agent._build_evidence_summary(sample_tracing_results)
        assert "drift=0.30" in summary

    def test_truncates_long_claim_text(self):
        """Claims over 200 chars are truncated in evidence summary."""
        long_claim = EvaluatedClaim(
            claim_text="x" * 300,
            source_paper_id="p1", source_title="Paper",
            depth=0, validity_score=0.8,
        )
        results = [TracingResult(
            paper_id="p1", paper_title="Paper", depth=0, claims=[long_claim]
        )]
        config = {"_groq_client": None}
        agent = ChallengerAgent(config)
        summary = agent._build_evidence_summary(results)
        # Should be truncated at 200 chars
        assert "x" * 201 not in summary
