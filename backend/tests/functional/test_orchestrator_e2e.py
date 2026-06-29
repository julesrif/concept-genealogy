"""
Functional test: end-to-end orchestrator run with all external I/O mocked.

Verifies the compiled LangGraph StateGraph executes the full pipeline
sequence from retrieve → persist with state propagation between all nodes.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.state import (
    PipelineState, PipelineStage, CandidatePaper,
    TracingResult, EvaluatedClaim, ExtractedClaim,
)


@pytest.fixture
def e2e_config():
    """Config for end-to-end test with no real API keys."""
    return {
        "groq_api_key": "",
        "groq_model_fast": "llama-3.1-8b-instant",
        "groq_model_reasoning": "llama-3.3-70b-versatile",
        "max_candidates": 3,
        "max_depth": 1,       # shallow for test speed
        "papers_per_level": 1,
        "_groq_client": None,
    }


@pytest.fixture
def mock_seed():
    return CandidatePaper(
        paper_id="seed_001",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        year=2017,
        citation_count=120000,
        open_access_url="https://arxiv.org/pdf/1706.03762.pdf",
        abstract="Dominant sequence transduction models...",
        doi="10.48550/arXiv.1706.03762",
    )


@pytest.fixture
def mock_ref_paper():
    return CandidatePaper(
        paper_id="ref_001",
        title="Neural Machine Translation by Jointly Learning to Align and Translate",
        authors=["Bahdanau"],
        year=2014,
        citation_count=30000,
        doi="10.48550/arXiv.1409.0473",
    )


class TestOrchestratorEndToEnd:
    """Full pipeline E2E test with compiled graph, all I/O mocked."""

    def test_concept_mode_happy_path(self, e2e_config, mock_seed, mock_ref_paper):
        """Concept mode: retrieve → gate1 → trace → graph_review → synth → gate2 → challenge → gate3 → persist."""
        with patch("core.groq_client.RateLimitedGroqClient"):
            from core.orchestrator import Orchestrator
            orch = Orchestrator(e2e_config)

        # -- Mock Retriever --
        orch.retriever.run = MagicMock(return_value=[mock_seed])
        orch.retriever.get_references = MagicMock(return_value=["ref_001"])
        orch.retriever.fetch_paper_by_id = MagicMock(return_value=mock_ref_paper)
        orch.retriever.fetch_paper_by_doi = MagicMock(return_value=mock_seed)
        orch.retriever.validate_doi = MagicMock(return_value=True)
        orch.retriever._rank_by_sjr = MagicMock(return_value=[mock_ref_paper])

        # -- Mock Extractor --
        seed_claims = [
            ExtractedClaim(
                claim_text="Transformer relies on attention.",
                source_paper_id="seed_001",
                source_title="Attention Is All You Need",
                depth=0,
            )
        ]
        ref_claims = [
            ExtractedClaim(
                claim_text="Alignment model for translation.",
                source_paper_id="ref_001",
                source_title="Neural Machine Translation",
                depth=1,
            )
        ]
        orch.extractor.run = MagicMock(side_effect=[seed_claims, ref_claims])

        # -- Mock Evaluator --
        seed_evaluated = [
            EvaluatedClaim(
                claim_text="Transformer relies on attention.",
                source_paper_id="seed_001",
                source_title="Attention Is All You Need",
                depth=0,
                validity_score=0.9,
                relevant_concepts=["attention mechanism"],
            )
        ]
        ref_evaluated = [
            EvaluatedClaim(
                claim_text="Alignment model for translation.",
                source_paper_id="ref_001",
                source_title="Neural Machine Translation",
                depth=1,
                validity_score=0.85,
                relevant_concepts=["attention"],
            )
        ]
        orch.evaluator.run = MagicMock(side_effect=[seed_evaluated, ref_evaluated])

        # -- Mock Inspector --
        seed_result = TracingResult(
            paper_id="seed_001",
            paper_title="Attention Is All You Need",
            depth=0,
            claims=seed_evaluated,
            references_explored=["ref_001"],
        )
        ref_result = TracingResult(
            paper_id="ref_001",
            paper_title="Neural Machine Translation",
            depth=1,
            claims=ref_evaluated,
            references_explored=[],
        )
        orch.inspector.validate_candidates = MagicMock(return_value=[mock_seed])
        orch.inspector.validate_tracing_result = MagicMock(
            side_effect=[seed_result, ref_result]
        )
        orch.inspector.validate_synthesis = MagicMock(
            return_value="Grounded synthesis of attention mechanism."
        )

        # -- Mock Synthesiser --
        orch.synthesiser.run = MagicMock(
            return_value="Grounded synthesis of attention mechanism."
        )

        # -- Mock Challenger --
        orch.challenger.run = MagicMock(return_value="No major issues found.")
        orch.challenger.last_issues = []
        orch.challenger.last_recommendation = "accept"

        # -- Mock HITL gates (auto-approve) --
        orch.hitl.gate_1_seed_selection = MagicMock(return_value=(True, mock_seed))
        orch.hitl.gate_graph_review = MagicMock(return_value=True)
        orch.hitl.gate_2_synthesis_approval = MagicMock(return_value=True)
        orch.hitl.gate_3_final_approval = MagicMock(return_value=True)

        # -- Mock Persistence --
        orch.persistence.save = MagicMock(return_value="output/result.json")

        # -- Execute --
        result = orch.run("attention mechanism")

        # -- Assertions --
        assert result.stage == PipelineStage.DONE
        assert result.output_path == "output/result.json"
        assert result.synthesis_text == "Grounded synthesis of attention mechanism."
        assert result.hitl_gate_1_approved is True
        assert result.hitl_graph_approved is True
        assert result.hitl_gate_2_approved is True
        assert result.hitl_gate_3_approved is True
        assert len(result.tracing_results) == 2
        # Verify agents were called
        orch.retriever.run.assert_called_once()
        assert orch.extractor.run.call_count == 2  # seed + ref
        orch.synthesiser.run.assert_called_once()
        orch.challenger.run.assert_called_once()
        orch.persistence.save.assert_called_once()
