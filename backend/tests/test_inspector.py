"""
tests/test_inspector.py - Unit tests for the Inspector agent.

Tests:
  - validate_candidates: accepts valid CandidatePaper objects and dicts
  - validate_candidates: discards malformed entries
  - validate_tracing_result: assembles TracingResult with depth correction
  - validate_synthesis: handles normal text, empty text, and JSON error responses
  - validate_extracted_claims_json: handles well-formed JSON, code fences, NDJSON
  - _clean_json_string: strips code fences

Implementation plan coverage: Week 2 (Inspector validation tests)
"""

import json

import pytest
from pydantic import ValidationError

from agents.inspector import InspectorAgent
from core.state import (
    PipelineState,
    CandidatePaper,
    EvaluatedClaim,
    TracingResult,
)


@pytest.fixture
def inspector(base_config):
    return InspectorAgent(base_config)


@pytest.fixture
def state():
    return PipelineState(user_concept="attention mechanism")


class TestValidateCandidates:
    """Tests for Inspector.validate_candidates()."""

    def test_accepts_valid_candidate_papers(self, inspector, state, sample_candidate_paper):
        """Already-typed CandidatePaper objects pass through."""
        result = inspector.validate_candidates([sample_candidate_paper], state)
        assert len(result) == 1
        assert result[0].paper_id == sample_candidate_paper.paper_id

    def test_accepts_valid_dict(self, inspector, state):
        """Raw dicts with correct fields are validated into CandidatePaper."""
        raw = {
            "paper_id": "test_id",
            "title": "Test Paper",
            "authors": ["Author One"],
            "year": 2023,
            "citation_count": 100,
        }
        result = inspector.validate_candidates([raw], state)
        assert len(result) == 1
        assert result[0].title == "Test Paper"

    def test_discards_malformed_entries(self, inspector, state):
        """Invalid entries are discarded, valid ones are kept."""
        valid = {"paper_id": "good", "title": "Good Paper", "authors": []}
        invalid_no_id = {"title": "No ID", "authors": []}  # missing paper_id
        invalid_type = "not a dict"

        result = inspector.validate_candidates([valid, invalid_no_id, invalid_type], state)
        # Only the valid entry should survive
        assert len(result) == 1
        assert result[0].paper_id == "good"

    def test_empty_list_returns_empty(self, inspector, state):
        """Empty input returns empty output."""
        result = inspector.validate_candidates([], state)
        assert result == []


class TestValidateTracingResult:
    """Tests for Inspector.validate_tracing_result()."""

    def test_assembles_tracing_result(
        self, inspector, sample_evaluated_claims, sample_candidate_paper, state
    ):
        """Assembles a valid TracingResult from claims and paper metadata."""
        result = inspector.validate_tracing_result(
            sample_evaluated_claims, sample_candidate_paper, depth=0, state=state,
        )
        assert isinstance(result, TracingResult)
        assert result.paper_id == sample_candidate_paper.paper_id
        assert result.depth == 0
        assert len(result.claims) == 2

    def test_corrects_depth_mismatch(
        self, inspector, sample_candidate_paper, state
    ):
        """Claims with wrong depth are silently corrected."""
        claims = [
            EvaluatedClaim(
                claim_text="Mismatched claim",
                source_paper_id="p1",
                source_title="T",
                depth=5,  # wrong - should be 0
                validity_score=0.8,
            )
        ]
        result = inspector.validate_tracing_result(
            claims, sample_candidate_paper, depth=0, state=state,
        )
        assert result.claims[0].depth == 0  # corrected

    def test_empty_claims_returns_valid_result(
        self, inspector, sample_candidate_paper, state
    ):
        """Empty claim list still produces a valid TracingResult."""
        result = inspector.validate_tracing_result(
            [], sample_candidate_paper, depth=1, state=state,
        )
        assert result.claims == []
        assert result.depth == 1


class TestValidateSynthesis:
    """Tests for Inspector.validate_synthesis()."""

    def test_valid_text_passes_through(self, inspector, state):
        """Normal Markdown text passes validation unchanged."""
        text = "# Genealogy Report\n\nThe concept evolved through three phases..."
        result = inspector.validate_synthesis(text, state)
        assert result == text

    def test_empty_string_returns_failure_message(self, inspector, state):
        """Empty synthesis is flagged."""
        result = inspector.validate_synthesis("", state)
        assert "synthesis failed" in result

    def test_json_error_response_detected(self, inspector, state):
        """LLM error JSON is caught and reported."""
        error_json = '{"error": "rate limit exceeded"}'
        result = inspector.validate_synthesis(error_json, state)
        assert "rate limit" in result

    def test_non_string_converted(self, inspector, state):
        """Non-string input is converted to string."""
        result = inspector.validate_synthesis(12345, state)
        assert result == "12345"


class TestValidateExtractedClaimsJson:
    """Tests for Inspector.validate_extracted_claims_json()."""

    def test_valid_json_array(self, inspector):
        """Well-formed JSON array of claims is parsed correctly."""
        raw = json.dumps([
            {"claim_text": "Claim 1", "inline_citations": [], "confidence": 0.9},
            {"claim_text": "Claim 2", "inline_citations": ["Doe, 2020"], "confidence": 0.8},
        ])
        result = inspector.validate_extracted_claims_json(raw, "paper_1", depth=0)
        assert len(result) == 2
        assert result[0]["claim_text"] == "Claim 1"

    def test_code_fenced_json(self, inspector):
        """JSON wrapped in ```json code fences is cleaned and parsed."""
        raw = '```json\n[{"claim_text": "Fenced claim", "confidence": 0.9}]\n```'
        result = inspector.validate_extracted_claims_json(raw, "paper_1", depth=0)
        assert len(result) == 1
        assert result[0]["claim_text"] == "Fenced claim"

    def test_single_object_wrapped(self, inspector):
        """A single JSON object (not array) is wrapped into an array."""
        raw = '{"claim_text": "Single claim", "confidence": 0.7}'
        result = inspector.validate_extracted_claims_json(raw, "paper_1", depth=0)
        assert len(result) == 1

    def test_ndjson_recovery(self, inspector):
        """NDJSON (one object per line) is recovered via line-by-line parsing."""
        raw = (
            '{"claim_text": "Claim A", "confidence": 0.9}\n'
            '{"claim_text": "Claim B", "confidence": 0.8}\n'
            '{"claim_text": "Claim C", "confidence": 0.7}'
        )
        result = inspector.validate_extracted_claims_json(raw, "paper_1", depth=0)
        assert len(result) == 3

    def test_invalid_json_returns_empty(self, inspector):
        """Completely malformed JSON returns empty list (triggers retry)."""
        raw = "This is not JSON at all"
        result = inspector.validate_extracted_claims_json(raw, "paper_1", depth=0)
        assert result == []

    def test_golden_fixture_parses(self, inspector, golden_fixture_claims_json):
        """Golden fixture (known-good LLM output) parses successfully."""
        result = inspector.validate_extracted_claims_json(
            golden_fixture_claims_json, "golden_paper", depth=0
        )
        assert len(result) == 5
        assert all("claim_text" in c for c in result)
        assert all("confidence" in c for c in result)


class TestCleanJsonString:
    """Tests for the _clean_json_string static method."""

    def test_strips_code_fences(self):
        raw = '```json\n[{"key": "value"}]\n```'
        cleaned = InspectorAgent._clean_json_string(raw)
        assert cleaned == '[{"key": "value"}]'

    def test_strips_plain_fences(self):
        raw = '```\n{"key": "value"}\n```'
        cleaned = InspectorAgent._clean_json_string(raw)
        assert cleaned == '{"key": "value"}'

    def test_no_change_for_clean_json(self):
        raw = '[{"claim": "test"}]'
        cleaned = InspectorAgent._clean_json_string(raw)
        assert cleaned == raw
