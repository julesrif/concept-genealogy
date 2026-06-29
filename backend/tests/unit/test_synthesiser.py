"""
Unit tests for agents/synthesiser.py - Synthesiser agent.

Same mock pattern as test_challenger.py / test_evaluator.py:
MagicMock Groq client with controllable chat_completion responses.
"""

import pytest
from unittest.mock import MagicMock

from agents.synthesiser import SynthesiserAgent, SYNTHESIS_PROMPT
from core.state import TracingResult, EvaluatedClaim


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def config_with_groq(mock_groq_client):
    return {
        "groq_model_reasoning": "llama-3.3-70b-versatile",
        "_groq_client": mock_groq_client,
    }


@pytest.fixture
def config_without_groq():
    return {
        "groq_model_reasoning": "llama-3.3-70b-versatile",
        "_groq_client": None,
    }


@pytest.fixture
def tracing_results_with_claims():
    """TracingResult list with valid claims at two depths."""
    return [
        TracingResult(
            paper_id="paper_001",
            paper_title="Attention Is All You Need",
            depth=0,
            claims=[
                EvaluatedClaim(
                    claim_text="The Transformer relies on attention mechanisms.",
                    source_paper_id="paper_001",
                    source_title="Attention Is All You Need",
                    depth=0,
                    validity_score=0.95,
                    relevant_concepts=["attention mechanism"],
                ),
                EvaluatedClaim(
                    claim_text="Multi-head attention enables joint attention.",
                    source_paper_id="paper_001",
                    source_title="Attention Is All You Need",
                    depth=0,
                    validity_score=0.88,
                    relevant_concepts=["multi-head attention"],
                ),
            ],
        ),
        TracingResult(
            paper_id="paper_002",
            paper_title="Neural Machine Translation by Jointly Learning",
            depth=1,
            claims=[
                EvaluatedClaim(
                    claim_text="Alignment model attends to source positions.",
                    source_paper_id="paper_002",
                    source_title="Neural Machine Translation by Jointly Learning",
                    depth=1,
                    validity_score=0.82,
                    relevant_concepts=["attention"],
                    drift_score=0.3,
                ),
            ],
        ),
    ]


@pytest.fixture
def tracing_results_no_claims():
    """TracingResult with no claims (empty list)."""
    return [
        TracingResult(
            paper_id="paper_003",
            paper_title="Empty Paper",
            depth=0,
            claims=[],
        ),
    ]


# ── Initialization ──────────────────────────────────────────────────────────

class TestSynthesiserInit:
    """Test SynthesiserAgent initialization."""

    def test_init_with_groq_client(self, config_with_groq):
        agent = SynthesiserAgent(config_with_groq)
        assert agent.groq is not None
        assert agent.model == "llama-3.3-70b-versatile"

    def test_init_without_groq_client(self, config_without_groq):
        agent = SynthesiserAgent(config_without_groq)
        assert agent.groq is None

    def test_init_default_model(self):
        agent = SynthesiserAgent({"_groq_client": None})
        assert agent.model == "llama-3.3-70b-versatile"


# ── Run method ──────────────────────────────────────────────────────────────

class TestSynthesiserRun:
    """Test the main run() method."""

    def test_happy_path_returns_synthesis(
        self, config_with_groq, tracing_results_with_claims
    ):
        """Valid claims + Groq client → returns LLM synthesis."""
        config_with_groq["_groq_client"].chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="A grounded synthesis."))]
        )
        agent = SynthesiserAgent(config_with_groq)
        result = agent.run(tracing_results_with_claims, "attention mechanism")
        assert result == "A grounded synthesis."
        config_with_groq["_groq_client"].chat_completion.assert_called_once()

    def test_empty_tracing_results_returns_skip(self, config_with_groq):
        """No tracing results → skip message."""
        agent = SynthesiserAgent(config_with_groq)
        result = agent.run([], "attention mechanism")
        assert "synthesis skipped" in result

    def test_zero_claims_returns_failure_message(
        self, config_with_groq, tracing_results_no_claims
    ):
        """Tracing results exist but all have 0 claims → failure message."""
        agent = SynthesiserAgent(config_with_groq)
        result = agent.run(tracing_results_no_claims, "attention mechanism")
        assert "Synthesis Failed" in result
        assert "No Claims Extracted" in result

    def test_feedback_triggers_revision_prompt(
        self, config_with_groq, tracing_results_with_claims
    ):
        """When feedback is provided, prompt includes revision instructions."""
        config_with_groq["_groq_client"].chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Revised synthesis."))]
        )
        agent = SynthesiserAgent(config_with_groq)
        result = agent.run(
            tracing_results_with_claims,
            "attention mechanism",
            feedback="Too many unsupported claims.",
        )
        # Verify the prompt sent to Groq includes revision text
        call_kwargs = config_with_groq["_groq_client"].chat_completion.call_args[1]
        prompt_content = call_kwargs["messages"][0]["content"]
        assert "REVISION REQUIRED" in prompt_content
        assert "Too many unsupported claims." in prompt_content

    def test_groq_exception_returns_error_string(
        self, config_with_groq, tracing_results_with_claims
    ):
        """LLM exception → returns error message, doesn't crash."""
        config_with_groq["_groq_client"].chat_completion.side_effect = RuntimeError(
            "Groq timeout"
        )
        agent = SynthesiserAgent(config_with_groq)
        result = agent.run(tracing_results_with_claims, "attention mechanism")
        assert "(synthesis failed:" in result
        assert "Groq timeout" in result


# ── Fallback without Groq ───────────────────────────────────────────────────

class TestSynthesiserFallback:
    """Test placeholder synthesis when Groq is unavailable."""

    def test_no_groq_returns_placeholder(
        self, config_without_groq, tracing_results_with_claims
    ):
        agent = SynthesiserAgent(config_without_groq)
        result = agent.run(tracing_results_with_claims, "attention mechanism")
        assert "placeholder" in result.lower()
        assert "attention mechanism" in result
        assert "LLM unavailable" in result

    def test_placeholder_includes_paper_titles(
        self, config_without_groq, tracing_results_with_claims
    ):
        agent = SynthesiserAgent(config_without_groq)
        result = agent.run(tracing_results_with_claims, "attention mechanism")
        assert "Attention Is All You Need" in result


# ── Static helpers ──────────────────────────────────────────────────────────

class TestFormatClaimsByDepth:
    """Test _format_claims_by_depth static method."""

    def test_groups_by_depth_and_assigns_ref_numbers(self, tracing_results_with_claims):
        result = SynthesiserAgent._format_claims_by_depth(tracing_results_with_claims)
        assert "[1]" in result  # first paper ref number
        assert "[2]" in result  # second paper ref number
        assert "Seed paper" in result
        assert "Depth 1" in result

    def test_empty_claims_paper_excluded(self):
        """Papers with no claims are not included."""
        results = [
            TracingResult(
                paper_id="empty",
                paper_title="Empty Paper",
                depth=0,
                claims=[],
            ),
            TracingResult(
                paper_id="full",
                paper_title="Full Paper",
                depth=1,
                claims=[
                    EvaluatedClaim(
                        claim_text="A claim.",
                        source_paper_id="full",
                        source_title="Full Paper",
                        depth=1,
                        validity_score=0.8,
                    )
                ],
            ),
        ]
        formatted = SynthesiserAgent._format_claims_by_depth(results)
        assert "Empty Paper" not in formatted
        assert "Full Paper" in formatted

    def test_no_results_returns_no_claims(self):
        formatted = SynthesiserAgent._format_claims_by_depth([])
        assert formatted == "(no claims)"


class TestPlaceholderSynthesis:
    """Test _placeholder_synthesis static method."""

    def test_placeholder_includes_stats(self, tracing_results_with_claims):
        result = SynthesiserAgent._placeholder_synthesis(
            tracing_results_with_claims, "attention mechanism"
        )
        assert "attention mechanism" in result
        assert "2" in result  # 2 papers
        assert "3" in result  # 3 total claims
