"""
tests/test_state.py - Unit tests for the pipeline state schema.

Validates:
  - PipelineState instantiation and defaults
  - PipelineStage enum values match the routing expectations
  - CandidatePaper, ExtractedClaim, EvaluatedClaim, TracingResult schemas
  - Pydantic validation rejects malformed data
  - Field constraints (ge, le) are enforced

Implementation plan coverage: Week 1 (state schema validation)
"""

import pytest
from pydantic import ValidationError

from core.state import (
    PipelineStage,
    PipelineState,
    CandidatePaper,
    ExtractedClaim,
    EvaluatedClaim,
    TracingResult,
)


class TestPipelineStage:
    """Tests for the PipelineStage enum."""

    def test_all_expected_stages_exist(self):
        """Verify all documented pipeline stages are present in enum."""
        expected = {
            "init", "validating_doi", "extracting_concepts",
            "awaiting_concept_selection", "retrieving",
            "awaiting_seed_approval", "tracing",
            "awaiting_graph_approval", "synthesising",
            "awaiting_synthesis_approval", "challenging",
            "awaiting_final_approval", "persisting",
            "done", "failed",
        }
        actual = {s.value for s in PipelineStage}
        assert actual == expected

    def test_stage_is_string_enum(self):
        """PipelineStage values can be used as string comparisons."""
        assert PipelineStage.INIT == "init"
        assert PipelineStage.DONE == "done"
        assert PipelineStage.FAILED == "failed"


class TestPipelineState:
    """Tests for the PipelineState global state model."""

    def test_default_initialisation(self):
        """PipelineState can be instantiated with no arguments (all defaults)."""
        state = PipelineState()
        assert state.stage == PipelineStage.INIT
        assert state.user_concept == ""
        assert state.max_depth == 3
        assert state.max_candidates == 3
        assert state.papers_per_level == 3
        assert state.current_depth == 0
        assert state.tracing_results == []
        assert state.candidate_papers == []
        assert state.visited_paper_ids == set()

    def test_doi_mode_fields(self):
        """DOI-mode specific fields initialise correctly."""
        state = PipelineState(doi_mode=True, input_doi="10.1234/test")
        assert state.doi_mode is True
        assert state.input_doi == "10.1234/test"
        assert state.extracted_concepts == []
        assert state.selected_concepts == []

    def test_retry_constraints(self):
        """Default retry limits match design proposal Section 5."""
        state = PipelineState()
        assert state.max_inspector_retries == 1
        assert state.max_synthesiser_retries == 3

    def test_hitl_gates_default_false(self):
        """All HITL gates start unapproved."""
        state = PipelineState()
        assert state.hitl_gate_1_approved is False
        assert state.hitl_graph_approved is False
        assert state.hitl_gate_2_approved is False
        assert state.hitl_gate_3_approved is False

    def test_visited_paper_ids_is_set(self):
        """visited_paper_ids supports set operations (deduplication)."""
        state = PipelineState()
        state.visited_paper_ids.add("paper_1")
        state.visited_paper_ids.add("paper_1")
        assert len(state.visited_paper_ids) == 1

    def test_concept_at_depth_dict(self):
        """concept_at_depth stores depth-specific concept abstraction."""
        state = PipelineState(
            concept_at_depth={0: "attention mechanism", 1: "selective attention"}
        )
        assert state.concept_at_depth[0] == "attention mechanism"
        assert state.concept_at_depth[1] == "selective attention"


class TestCandidatePaper:
    """Tests for the CandidatePaper schema."""

    def test_minimal_valid_paper(self):
        """CandidatePaper with only required fields."""
        paper = CandidatePaper(
            paper_id="abc123",
            title="Test Paper",
            authors=["Author One"],
        )
        assert paper.paper_id == "abc123"
        assert paper.citation_count == 0
        assert paper.sjr_score is None
        assert paper.open_access_url is None

    def test_full_paper(self, sample_candidate_paper):
        """CandidatePaper with all fields populated."""
        assert sample_candidate_paper.paper_id == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        assert sample_candidate_paper.year == 2017
        assert sample_candidate_paper.doi == "10.48550/arXiv.1706.03762"
        assert len(sample_candidate_paper.authors) == 3

    def test_missing_required_field_raises(self):
        """Missing paper_id raises ValidationError."""
        with pytest.raises(ValidationError):
            CandidatePaper(title="No ID", authors=[])


class TestExtractedClaim:
    """Tests for the ExtractedClaim schema."""

    def test_valid_claim(self):
        """ExtractedClaim with all fields."""
        claim = ExtractedClaim(
            claim_text="Some academic claim.",
            source_paper_id="paper_1",
            source_title="Paper Title",
            depth=1,
            inline_citations=["Doe, 2020"],
            confidence=0.9,
        )
        assert claim.depth == 1
        assert claim.confidence == 0.9

    def test_confidence_bounds(self):
        """confidence field is constrained to [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            ExtractedClaim(
                claim_text="x", source_paper_id="y",
                source_title="z", depth=0, confidence=1.5,
            )
        with pytest.raises(ValidationError):
            ExtractedClaim(
                claim_text="x", source_paper_id="y",
                source_title="z", depth=0, confidence=-0.1,
            )

    def test_default_confidence(self):
        """Default confidence is 1.0."""
        claim = ExtractedClaim(
            claim_text="x", source_paper_id="y",
            source_title="z", depth=0,
        )
        assert claim.confidence == 1.0


class TestEvaluatedClaim:
    """Tests for the EvaluatedClaim schema."""

    def test_valid_evaluated_claim(self):
        claim = EvaluatedClaim(
            claim_text="A valid claim.",
            source_paper_id="p1",
            source_title="Title",
            depth=0,
            validity_score=0.85,
            relevant_concepts=["concept_a"],
        )
        assert claim.validity_score == 0.85
        assert claim.drift_score is None

    def test_validity_score_bounds(self):
        """validity_score must be in [0, 1]."""
        with pytest.raises(ValidationError):
            EvaluatedClaim(
                claim_text="x", source_paper_id="y",
                source_title="z", depth=0, validity_score=2.0,
            )

    def test_drift_score_optional(self):
        """drift_score is None at depth 0 (no parent to drift from)."""
        claim = EvaluatedClaim(
            claim_text="x", source_paper_id="y",
            source_title="z", depth=0, validity_score=0.5,
        )
        assert claim.drift_score is None

    def test_drift_score_at_depth_1(self):
        """drift_score can be set at depth > 0."""
        claim = EvaluatedClaim(
            claim_text="x", source_paper_id="y",
            source_title="z", depth=1, validity_score=0.7,
            drift_score=0.3,
        )
        assert claim.drift_score == 0.3


class TestTracingResult:
    """Tests for the TracingResult schema."""

    def test_valid_tracing_result(self, sample_evaluated_claims, sample_candidate_paper):
        """TracingResult assembles correctly from evaluated claims."""
        result = TracingResult(
            paper_id=sample_candidate_paper.paper_id,
            paper_title=sample_candidate_paper.title,
            depth=0,
            authors=sample_candidate_paper.authors,
            doi=sample_candidate_paper.doi,
            claims=sample_evaluated_claims,
        )
        assert result.depth == 0
        assert len(result.claims) == 2
        assert result.doi == "10.48550/arXiv.1706.03762"

    def test_empty_claims_allowed(self):
        """TracingResult with no claims is valid (paper may have no extractable claims)."""
        result = TracingResult(
            paper_id="p1",
            paper_title="Empty Paper",
            depth=2,
        )
        assert result.claims == []
        assert result.references_explored == []
