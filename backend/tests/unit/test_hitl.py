"""
Unit tests for agents/hitl.py - Human-in-the-Loop gate implementations.

Tests all 5 gate methods with mocked input() to simulate user interaction.
"""

import pytest
from unittest.mock import patch, call

from agents.hitl import HITLGate
from core.state import CandidatePaper


@pytest.fixture
def gate():
    return HITLGate()


@pytest.fixture
def candidates():
    return [
        CandidatePaper(
            paper_id="p1",
            title="Paper One",
            authors=["Author A"],
            year=2020,
            citation_count=100,
            open_access_url="https://example.com/p1.pdf",
            abstract="Abstract one",
        ),
        CandidatePaper(
            paper_id="p2",
            title="Paper Two",
            authors=["Author B", "Author C", "Author D", "Author E"],
            year=2019,
            citation_count=50,
        ),
    ]


# ── Gate 1: Seed Selection ──────────────────────────────────────────────────

class TestGate1SeedSelection:
    """Test gate_1_seed_selection with various user inputs."""

    def test_select_first_candidate(self, gate, candidates):
        with patch("builtins.input", return_value="1"):
            approved, selected = gate.gate_1_seed_selection(candidates)
        assert approved is True
        assert selected == candidates[0]

    def test_select_second_candidate(self, gate, candidates):
        with patch("builtins.input", return_value="2"):
            approved, selected = gate.gate_1_seed_selection(candidates)
        assert approved is True
        assert selected == candidates[1]

    def test_reject_all_candidates(self, gate, candidates):
        with patch("builtins.input", return_value="r"):
            approved, selected = gate.gate_1_seed_selection(candidates)
        assert approved is False
        assert selected is None

    def test_quit_raises_system_exit(self, gate, candidates):
        with patch("builtins.input", return_value="q"):
            with pytest.raises(SystemExit):
                gate.gate_1_seed_selection(candidates)

    def test_non_interactive_auto_selects_first(self, gate, candidates):
        """EOFError simulates non-interactive mode (CI/tests)."""
        with patch("builtins.input", side_effect=EOFError):
            approved, selected = gate.gate_1_seed_selection(candidates)
        assert approved is True
        assert selected == candidates[0]

    def test_invalid_then_valid_input(self, gate, candidates):
        """Invalid input is retried until valid."""
        with patch("builtins.input", side_effect=["x", "99", "1"]):
            approved, selected = gate.gate_1_seed_selection(candidates)
        assert approved is True
        assert selected == candidates[0]

    def test_empty_candidates_non_interactive(self, gate):
        """Non-interactive with empty candidates returns (False, None)."""
        with patch("builtins.input", side_effect=EOFError):
            approved, selected = gate.gate_1_seed_selection([])
        assert approved is False
        assert selected is None


# ── Gate 2: Synthesis Approval ──────────────────────────────────────────────

class TestGate2SynthesisApproval:
    """Test gate_2_synthesis_approval."""

    def test_approve_synthesis(self, gate):
        with patch("builtins.input", return_value="a"):
            result = gate.gate_2_synthesis_approval("A synthesis text.")
        assert result is True

    def test_reject_synthesis(self, gate):
        with patch("builtins.input", return_value="r"):
            result = gate.gate_2_synthesis_approval("A synthesis text.")
        assert result is False

    def test_quit_raises_system_exit(self, gate):
        with patch("builtins.input", return_value="q"):
            with pytest.raises(SystemExit):
                gate.gate_2_synthesis_approval("text")

    def test_non_interactive_auto_approves(self, gate):
        with patch("builtins.input", side_effect=EOFError):
            result = gate.gate_2_synthesis_approval("text")
        assert result is True

    def test_none_synthesis_displays_empty(self, gate):
        """None synthesis shows placeholder without crashing."""
        with patch("builtins.input", return_value="a"):
            result = gate.gate_2_synthesis_approval(None)
        assert result is True


# ── Gate Graph Review ───────────────────────────────────────────────────────

class TestGateGraphReview:
    """Test gate_graph_review."""

    def test_approve_graph(self, gate):
        with patch("builtins.input", return_value="a"):
            result = gate.gate_graph_review()
        assert result is True

    def test_reject_graph(self, gate):
        with patch("builtins.input", return_value="r"):
            result = gate.gate_graph_review()
        assert result is False

    def test_quit_raises_system_exit(self, gate):
        with patch("builtins.input", return_value="q"):
            with pytest.raises(SystemExit):
                gate.gate_graph_review()

    def test_non_interactive_auto_approves(self, gate):
        with patch("builtins.input", side_effect=EOFError):
            result = gate.gate_graph_review()
        assert result is True


# ── Gate Concept Selection (DOI Mode) ───────────────────────────────────────

class TestGateConceptSelection:
    """Test gate_concept_selection for DOI mode."""

    def test_select_single_concept(self, gate):
        concepts = ["attention mechanism", "transformer", "self-attention"]
        with patch("builtins.input", return_value="1"):
            result = gate.gate_concept_selection(concepts)
        assert result == ["attention mechanism"]

    def test_select_multiple_concepts(self, gate):
        concepts = ["attention mechanism", "transformer", "self-attention"]
        with patch("builtins.input", return_value="1,3"):
            result = gate.gate_concept_selection(concepts)
        assert result == ["attention mechanism", "self-attention"]

    def test_max_three_concepts(self, gate):
        concepts = ["a", "b", "c", "d", "e"]
        with patch("builtins.input", return_value="1,2,3,4,5"):
            result = gate.gate_concept_selection(concepts)
        # Only first 3 selected
        assert len(result) == 3
        assert result == ["a", "b", "c"]

    def test_reject_concepts(self, gate):
        concepts = ["attention mechanism"]
        with patch("builtins.input", return_value="r"):
            result = gate.gate_concept_selection(concepts)
        assert result is None

    def test_quit_raises_system_exit(self, gate):
        concepts = ["attention mechanism"]
        with patch("builtins.input", return_value="q"):
            with pytest.raises(SystemExit):
                gate.gate_concept_selection(concepts)

    def test_non_interactive_auto_selects_first(self, gate):
        concepts = ["attention mechanism", "transformer"]
        with patch("builtins.input", side_effect=EOFError):
            result = gate.gate_concept_selection(concepts)
        assert result == ["attention mechanism"]

    def test_invalid_then_valid_input(self, gate):
        concepts = ["a", "b", "c"]
        with patch("builtins.input", side_effect=["x", "2"]):
            result = gate.gate_concept_selection(concepts)
        assert result == ["b"]


# ── Gate 3: Final Approval ──────────────────────────────────────────────────

class TestGate3FinalApproval:
    """Test gate_3_final_approval."""

    def test_approve_final(self, gate):
        with patch("builtins.input", return_value="a"):
            result = gate.gate_3_final_approval("synthesis", "critique", [{"issue": "x"}])
        assert result is True

    def test_reject_final(self, gate):
        with patch("builtins.input", return_value="r"):
            result = gate.gate_3_final_approval("synthesis", "critique", [])
        assert result is False

    def test_quit_raises_system_exit(self, gate):
        with patch("builtins.input", return_value="q"):
            with pytest.raises(SystemExit):
                gate.gate_3_final_approval("synthesis", None, None)

    def test_non_interactive_auto_approves(self, gate):
        with patch("builtins.input", side_effect=EOFError):
            result = gate.gate_3_final_approval("synthesis", "critique", [])
        assert result is True

    def test_no_critique_displays_without_crash(self, gate):
        with patch("builtins.input", return_value="a"):
            result = gate.gate_3_final_approval("synthesis", None, None)
        assert result is True


# ── Shared _prompt_approve_reject ───────────────────────────────────────────

class TestPromptApproveReject:
    """Test the shared approve/reject helper."""

    def test_accepts_yes_variants(self, gate):
        for answer in ["a", "approve", "yes", "y"]:
            with patch("builtins.input", return_value=answer):
                assert HITLGate._prompt_approve_reject("test") is True

    def test_accepts_reject_variants(self, gate):
        for answer in ["r", "reject", "no", "n"]:
            with patch("builtins.input", return_value=answer):
                assert HITLGate._prompt_approve_reject("test") is False

    def test_invalid_then_valid(self, gate):
        with patch("builtins.input", side_effect=["garbage", "a"]):
            assert HITLGate._prompt_approve_reject("test") is True
