"""
tests/test_golden_fixture.py - Integration test using golden fixture data.

Tests the Inspector → Evaluator → TracingResult pipeline with known-good
golden fixture data (5 claims from "Attention Is All You Need"). No network
calls are made — all dependencies are mocked.

Implementation plan coverage: Week 3 (Integration / golden path tests)
"""

from unittest.mock import MagicMock

import pytest

from agents.inspector import InspectorAgent
from agents.evaluator import EvaluatorAgent
from core.state import (
    PipelineState,
    CandidatePaper,
    EvaluatedClaim,
    TracingResult,
    ExtractedClaim,
)


@pytest.fixture
def golden_paper():
    """The canonical seed paper for the golden fixture."""
    return CandidatePaper(
        paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        title="Attention Is All You Need",
        authors=["Vaswani, Ashish", "Shazeer, Noam", "Parmar, Niki"],
        year=2017,
        citation_count=120000,
        doi="10.48550/arXiv.1706.03762",
        open_access_url="https://arxiv.org/pdf/1706.03762.pdf",
    )


class TestGoldenFixturePipeline:
    """Integration tests exercising the Inspector → Evaluator → TracingResult flow."""

    def test_inspector_parses_golden_claims(
        self, base_config, golden_fixture_claims_json, golden_paper
    ):
        """Golden fixture JSON is successfully parsed by Inspector."""
        inspector = InspectorAgent(base_config)
        claims = inspector.validate_extracted_claims_json(
            golden_fixture_claims_json, golden_paper.paper_id, depth=0,
        )
        assert len(claims) == 5
        assert all("claim_text" in c for c in claims)
        assert all(c["confidence"] >= 0.7 for c in claims)

    def test_golden_claims_become_extracted_claims(
        self, base_config, golden_fixture_claims_json, golden_paper
    ):
        """Parsed golden claims can be converted to ExtractedClaim objects."""
        inspector = InspectorAgent(base_config)
        raw_claims = inspector.validate_extracted_claims_json(
            golden_fixture_claims_json, golden_paper.paper_id, depth=0,
        )

        typed_claims = []
        for c in raw_claims:
            typed_claims.append(ExtractedClaim(
                claim_text=c["claim_text"],
                source_paper_id=golden_paper.paper_id,
                source_title=golden_paper.title,
                depth=0,
                inline_citations=c.get("inline_citations", []),
                confidence=c.get("confidence", 1.0),
            ))
        assert len(typed_claims) == 5
        assert all(isinstance(c, ExtractedClaim) for c in typed_claims)

    def test_evaluator_scores_golden_claims(
        self, base_config, golden_fixture_claims_json, golden_paper
    ):
        """Evaluator assigns scores to golden claims (mocked LLM)."""
        # Mock LLM to return high validity for relevant claims
        mock_groq = MagicMock()
        mock_groq.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"relevant": true, "validity_score": 0.9, "relevant_concepts": ["attention"]}'
            ))]
        )

        inspector = InspectorAgent(base_config)
        raw_claims = inspector.validate_extracted_claims_json(
            golden_fixture_claims_json, golden_paper.paper_id, depth=0,
        )

        typed_claims = [
            ExtractedClaim(
                claim_text=c["claim_text"],
                source_paper_id=golden_paper.paper_id,
                source_title=golden_paper.title,
                depth=0,
                confidence=c.get("confidence", 1.0),
            )
            for c in raw_claims
        ]

        config_with_groq = {**base_config, "_groq_client": mock_groq}
        state = PipelineState(user_concept="attention mechanism")
        evaluator = EvaluatorAgent(config_with_groq)

        evaluated = evaluator.run(typed_claims, state)
        assert len(evaluated) >= 1
        assert all(isinstance(c, EvaluatedClaim) for c in evaluated)
        assert all(c.validity_score >= 0.0 for c in evaluated)

    def test_full_golden_pipeline_produces_tracing_result(
        self, base_config, golden_fixture_claims_json, golden_paper
    ):
        """Full golden pipeline: Inspector → Evaluator → TracingResult."""
        mock_groq = MagicMock()
        mock_groq.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"relevant": true, "validity_score": 0.85, "relevant_concepts": ["attention"]}'
            ))]
        )

        inspector = InspectorAgent(base_config)
        state = PipelineState(user_concept="attention mechanism")

        # Step 1: Inspector parses raw JSON
        raw_claims = inspector.validate_extracted_claims_json(
            golden_fixture_claims_json, golden_paper.paper_id, depth=0,
        )
        assert len(raw_claims) == 5

        # Step 2: Convert to typed claims
        typed_claims = [
            ExtractedClaim(
                claim_text=c["claim_text"],
                source_paper_id=golden_paper.paper_id,
                source_title=golden_paper.title,
                depth=0,
                confidence=c.get("confidence", 1.0),
            )
            for c in raw_claims
        ]

        # Step 3: Evaluator scores claims
        config_with_groq = {**base_config, "_groq_client": mock_groq}
        evaluator = EvaluatorAgent(config_with_groq)
        evaluated = evaluator.run(typed_claims, state)
        assert len(evaluated) >= 1

        # Step 4: Inspector assembles TracingResult
        tracing_result = inspector.validate_tracing_result(
            evaluated, golden_paper, depth=0, state=state,
        )
        assert isinstance(tracing_result, TracingResult)
        assert tracing_result.paper_id == golden_paper.paper_id
        assert tracing_result.paper_title == "Attention Is All You Need"
        assert tracing_result.depth == 0
        assert len(tracing_result.claims) >= 1

    def test_golden_claims_have_inline_citations(self, base_config, golden_fixture_claims_json):
        """At least some golden claims have inline citations (testing extraction quality)."""
        inspector = InspectorAgent(base_config)
        raw_claims = inspector.validate_extracted_claims_json(
            golden_fixture_claims_json, "test_paper", depth=0,
        )
        claims_with_citations = [
            c for c in raw_claims if c.get("inline_citations")
        ]
        # At least 2 of the 5 golden claims should have citations
        assert len(claims_with_citations) >= 2
