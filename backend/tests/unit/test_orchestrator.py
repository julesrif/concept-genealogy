"""
Unit tests for core/orchestrator.py - the central LangGraph StateGraph.

Tests graph construction, routing logic, node sequencing, retry caps,
and error handling. All agents are mocked at instance level.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core.state import (
    PipelineState, PipelineStage, CandidatePaper,
    TracingResult, EvaluatedClaim,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """Config dict that avoids real Groq SDK import."""
    return {
        "groq_api_key": "",
        "groq_model_fast": "llama-3.1-8b-instant",
        "groq_model_reasoning": "llama-3.3-70b-versatile",
        "max_candidates": 3,
        "max_depth": 2,
        "papers_per_level": 2,
        "_groq_client": None,
    }


@pytest.fixture
def orchestrator(minimal_config):
    """Orchestrator with mocked agents and no real Groq client."""
    with patch("core.groq_client.RateLimitedGroqClient"):
        from core.orchestrator import Orchestrator
        orch = Orchestrator(minimal_config)

    # Replace all agents with mocks
    orch.retriever = MagicMock()
    orch.extractor = MagicMock()
    orch.evaluator = MagicMock()
    orch.inspector = MagicMock()
    orch.synthesiser = MagicMock()
    orch.challenger = MagicMock()
    orch.hitl = MagicMock()
    orch.persistence = MagicMock()
    return orch


@pytest.fixture
def seed_paper():
    return CandidatePaper(
        paper_id="seed123",
        title="Test Seed Paper",
        authors=["Author A"],
        year=2020,
        citation_count=100,
        open_access_url="https://example.com/paper.pdf",
        abstract="Test abstract.",
        doi="10.1234/test",
    )


@pytest.fixture
def tracing_result_depth0():
    return TracingResult(
        paper_id="seed123",
        paper_title="Test Seed Paper",
        depth=0,
        authors=["Author A"],
        doi="10.1234/test",
        claims=[
            EvaluatedClaim(
                claim_text="Test claim",
                source_paper_id="seed123",
                source_title="Test Seed Paper",
                depth=0,
                validity_score=0.9,
                relevant_concepts=["test concept"],
            )
        ],
        references_explored=["ref1", "ref2", "ref3"],
    )


# ── Graph construction ──────────────────────────────────────────────────────

class TestGraphConstruction:
    """Verify the compiled graph has the expected nodes and edges."""

    def test_graph_has_all_expected_nodes(self, orchestrator):
        """The StateGraph must include all 12 pipeline nodes."""
        # LangGraph compiled graph stores nodes in .nodes
        graph = orchestrator.graph
        # Get node names from the compiled graph
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "validate_doi",
            "extract_concepts",
            "hitl_concept_selection",
            "retrieve",
            "hitl_gate_1",
            "trace_citation",
            "hitl_graph_review",
            "synthesise",
            "hitl_gate_2",
            "challenge",
            "hitl_gate_3",
            "persist",
            "__start__",  # LangGraph internal
            "__end__",    # LangGraph internal
        }
        # All pipeline nodes must be present
        pipeline_nodes = expected_nodes - {"__start__", "__end__"}
        for node in pipeline_nodes:
            assert node in node_names, f"Missing node: {node}"

    def test_graph_compiles_without_error(self, orchestrator):
        """Graph must compile (already done in __init__)."""
        assert orchestrator.graph is not None


# ── Routing functions ───────────────────────────────────────────────────────

class TestRouteAfterDoiValidation:
    """Test _route_after_doi_validation branching."""

    def test_doi_mode_routes_to_extract_concepts(self, orchestrator):
        state = {"doi_mode": True, "stage": PipelineStage.VALIDATING_DOI.value}
        assert orchestrator._route_after_doi_validation(state) == "extract_concepts"

    def test_doi_mode_failed_routes_to_end(self, orchestrator):
        state = {"doi_mode": True, "stage": PipelineStage.FAILED.value}
        from langgraph.graph import END
        assert orchestrator._route_after_doi_validation(state) == END

    def test_concept_mode_routes_to_retrieve(self, orchestrator):
        state = {"doi_mode": False}
        assert orchestrator._route_after_doi_validation(state) == "retrieve"


class TestRouteAfterTracing:
    """Test the BFS tracing routing logic."""

    def test_pending_papers_routes_to_trace_citation(self, orchestrator):
        state = {
            "current_depth": 1,
            "max_depth": 3,
            "visited_paper_ids": [],
            "pending_papers_at_depth": ["paper_a"],
            "tracing_results": [],
        }
        assert orchestrator._route_after_tracing(state) == "trace_citation"

    def test_depth_cap_reached_routes_to_graph_review(self, orchestrator):
        state = {
            "current_depth": 4,  # > max_depth=3
            "max_depth": 3,
            "visited_paper_ids": [],
            "pending_papers_at_depth": [],
            "tracing_results": [],
        }
        assert orchestrator._route_after_tracing(state) == "hitl_graph_review"

    def test_no_refs_after_seed_routes_to_gate_1(self, orchestrator):
        """When depth<=1 and no new refs found, route back to gate_1 (seed failed)."""
        state = {
            "current_depth": 1,
            "max_depth": 3,
            "visited_paper_ids": [],
            "pending_papers_at_depth": [],
            "tracing_results": [
                {"depth": 0, "references_explored": []},
            ],
        }
        result = orchestrator._route_after_tracing(state)
        assert result == "hitl_gate_1"

    def test_new_refs_available_routes_back_to_trace(self, orchestrator):
        """If there are unvisited refs from the last depth, continue tracing."""
        state = {
            "current_depth": 2,
            "max_depth": 3,
            "visited_paper_ids": ["seed123"],
            "pending_papers_at_depth": [],
            "tracing_results": [
                {"depth": 1, "references_explored": ["new_ref_1", "new_ref_2"]},
            ],
        }
        result = orchestrator._route_after_tracing(state)
        assert result == "trace_citation"

    def test_all_refs_visited_routes_to_graph_review(self, orchestrator):
        """All references already visited → stop tracing."""
        state = {
            "current_depth": 2,
            "max_depth": 3,
            "visited_paper_ids": ["ref1", "ref2"],
            "pending_papers_at_depth": [],
            "tracing_results": [
                {"depth": 1, "references_explored": ["ref1", "ref2"]},
            ],
        }
        result = orchestrator._route_after_tracing(state)
        assert result == "hitl_graph_review"


class TestRouteAfterSynthesisReview:
    """Test routing after HITL Gate 2."""

    def test_approved_routes_to_challenge(self, orchestrator):
        state = {"hitl_gate_2_approved": True}
        assert orchestrator._route_after_synthesis_review(state) == "challenge"

    def test_rejected_under_cap_routes_to_synthesise(self, orchestrator):
        state = {
            "hitl_gate_2_approved": False,
            "synthesiser_retries": 1,
            "max_synthesiser_retries": 3,
        }
        assert orchestrator._route_after_synthesis_review(state) == "synthesise"

    def test_rejected_at_cap_routes_to_persist(self, orchestrator):
        state = {
            "hitl_gate_2_approved": False,
            "synthesiser_retries": 3,
            "max_synthesiser_retries": 3,
        }
        result = orchestrator._route_after_synthesis_review(state)
        assert result == "persist"


class TestRouteAfterFinalReview:
    """Test routing after HITL Gate 3."""

    def test_approved_routes_to_persist(self, orchestrator):
        state = {"hitl_gate_3_approved": True}
        assert orchestrator._route_after_final_review(state) == "persist"

    def test_rejected_under_cap_routes_to_challenge(self, orchestrator):
        state = {
            "hitl_gate_3_approved": False,
            "challenger_retries": 1,
            "max_challenger_retries": 3,
        }
        assert orchestrator._route_after_final_review(state) == "challenge"

    def test_rejected_at_cap_routes_to_persist(self, orchestrator):
        state = {
            "hitl_gate_3_approved": False,
            "challenger_retries": 3,
            "max_challenger_retries": 3,
        }
        result = orchestrator._route_after_final_review(state)
        assert result == "persist"


# ── Node implementations ────────────────────────────────────────────────────

class TestNodeValidateDoi:
    """Test _node_validate_doi logic."""

    def test_concept_mode_passthrough(self, orchestrator):
        state = PipelineState(user_concept="test", doi_mode=False)
        result = orchestrator._node_validate_doi(state)
        assert result.stage == PipelineStage.INIT  # unchanged

    def test_doi_mode_no_doi_fails(self, orchestrator):
        state = PipelineState(doi_mode=True, input_doi=None)
        result = orchestrator._node_validate_doi(state)
        assert result.stage == PipelineStage.FAILED
        assert "DOI mode requires" in result.error_message

    def test_doi_mode_invalid_doi_fails(self, orchestrator):
        orchestrator.retriever.validate_doi.return_value = False
        state = PipelineState(doi_mode=True, input_doi="10.invalid/doi")
        result = orchestrator._node_validate_doi(state)
        assert result.stage == PipelineStage.FAILED
        assert "DOI not found" in result.error_message

    def test_doi_mode_valid_doi_sets_seed(self, orchestrator, seed_paper):
        orchestrator.retriever.validate_doi.return_value = True
        orchestrator.retriever.fetch_paper_by_doi.return_value = seed_paper
        state = PipelineState(doi_mode=True, input_doi="10.1234/test")
        result = orchestrator._node_validate_doi(state)
        assert result.selected_seed == seed_paper
        assert seed_paper.paper_id in result.visited_paper_ids


class TestNodeRetrieve:
    """Test _node_retrieve activates Retriever and Inspector."""

    def test_retrieve_calls_retriever_and_inspector(self, orchestrator, seed_paper):
        orchestrator.retriever.run.return_value = [seed_paper]
        orchestrator.inspector.validate_candidates.return_value = [seed_paper]
        state = PipelineState(user_concept="attention mechanism")
        result = orchestrator._node_retrieve(state)
        orchestrator.retriever.run.assert_called_once_with("attention mechanism")
        orchestrator.inspector.validate_candidates.assert_called_once()
        assert result.candidate_papers == [seed_paper]
        assert result.stage == PipelineStage.AWAITING_SEED_APPROVAL


class TestNodeHitlGate1:
    """Test seed selection gate logic."""

    def test_gate_1_approved(self, orchestrator, seed_paper):
        orchestrator.hitl.gate_1_seed_selection.return_value = (True, seed_paper)
        state = PipelineState(
            user_concept="test",
            candidate_papers=[seed_paper],
        )
        result = orchestrator._node_hitl_gate_1(state)
        assert result.hitl_gate_1_approved is True
        assert result.selected_seed == seed_paper
        assert seed_paper.paper_id in result.visited_paper_ids

    def test_gate_1_rejected(self, orchestrator, seed_paper):
        orchestrator.hitl.gate_1_seed_selection.return_value = (False, None)
        state = PipelineState(
            user_concept="test",
            candidate_papers=[seed_paper],
        )
        result = orchestrator._node_hitl_gate_1(state)
        assert result.hitl_gate_1_approved is False


class TestNodeTraceCitation:
    """Test the trace_citation node core loop."""

    def test_depth_0_traces_seed_paper(self, orchestrator, seed_paper):
        """At depth 0, traces the selected seed paper."""
        mock_claims = [MagicMock()]
        mock_evaluated = [MagicMock()]
        mock_result = TracingResult(
            paper_id=seed_paper.paper_id,
            paper_title=seed_paper.title,
            depth=0,
            claims=[
                EvaluatedClaim(
                    claim_text="claim",
                    source_paper_id=seed_paper.paper_id,
                    source_title=seed_paper.title,
                    depth=0,
                    validity_score=0.9,
                )
            ],
            references_explored=["ref1"],
        )
        orchestrator.extractor.run.return_value = mock_claims
        orchestrator.evaluator.run.return_value = mock_evaluated
        orchestrator.inspector.validate_tracing_result.return_value = mock_result
        orchestrator.retriever.get_references.return_value = ["ref1"]

        state = PipelineState(
            user_concept="test",
            selected_seed=seed_paper,
            current_depth=0,
        )
        result = orchestrator._node_trace_citation(state)
        assert result.current_depth == 1  # advanced after seed
        assert len(result.tracing_results) == 1
        assert seed_paper.paper_id in result.visited_paper_ids

    def test_deeper_depth_pops_from_queue(self, orchestrator, seed_paper):
        """At depth > 0, pops paper from pending queue."""
        ref_paper = CandidatePaper(
            paper_id="ref1",
            title="Reference Paper",
            authors=["B"],
            year=2018,
        )
        mock_result = TracingResult(
            paper_id="ref1",
            paper_title="Reference Paper",
            depth=1,
            claims=[],
            references_explored=[],
        )
        orchestrator.extractor.run.return_value = []
        orchestrator.evaluator.run.return_value = []
        orchestrator.inspector.validate_tracing_result.return_value = mock_result
        orchestrator.retriever.get_references.return_value = []
        orchestrator.retriever.fetch_paper_by_id.return_value = ref_paper

        state = PipelineState(
            user_concept="test",
            selected_seed=seed_paper,
            current_depth=1,
            pending_papers_at_depth=["ref1"],
            visited_paper_ids={"seed123"},
            concept_at_depth={0: "test"},
        )
        result = orchestrator._node_trace_citation(state)
        assert "ref1" in result.visited_paper_ids


class TestNodeSynthesise:
    """Test synthesis node."""

    def test_synthesise_calls_synthesiser(self, orchestrator, tracing_result_depth0):
        orchestrator.synthesiser.run.return_value = "A grounded synthesis."
        orchestrator.inspector.validate_synthesis.return_value = "A grounded synthesis."
        state = PipelineState(
            user_concept="test",
            tracing_results=[tracing_result_depth0],
        )
        result = orchestrator._node_synthesise(state)
        assert result.synthesis_text == "A grounded synthesis."
        assert result.stage == PipelineStage.AWAITING_SYNTHESIS_APPROVAL

    def test_synthesise_with_retry_includes_feedback(self, orchestrator, tracing_result_depth0):
        orchestrator.synthesiser.run.return_value = "Revised synthesis."
        orchestrator.inspector.validate_synthesis.return_value = "Revised synthesis."
        state = PipelineState(
            user_concept="test",
            tracing_results=[tracing_result_depth0],
            synthesiser_retries=1,
            challenger_critique="Too many unsupported claims.",
        )
        result = orchestrator._node_synthesise(state)
        # synthesiser.run should be called with feedback kwarg
        call_kwargs = orchestrator.synthesiser.run.call_args[1]
        assert call_kwargs["feedback"] == "Too many unsupported claims."


class TestNodeChallenge:
    """Test challenger node with auto-retry logic."""

    def test_challenge_runs_challenger(self, orchestrator, tracing_result_depth0):
        orchestrator.challenger.run.return_value = "Critique text"
        orchestrator.challenger.last_issues = [{"issue": "overclaim"}]
        orchestrator.challenger.last_recommendation = "accept"
        state = PipelineState(
            user_concept="test",
            synthesis_text="Some synthesis",
            tracing_results=[tracing_result_depth0],
        )
        result = orchestrator._node_challenge(state)
        assert result.challenger_critique == "Critique text"
        assert result.stage == PipelineStage.AWAITING_FINAL_APPROVAL

    def test_challenge_auto_retries_on_revise(self, orchestrator, tracing_result_depth0):
        """When challenger recommends 'revise', auto-retries up to 2 times."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return f"Critique #{call_count[0]}"

        orchestrator.challenger.run.side_effect = side_effect
        # First 2 calls return "revise", third returns "accept"
        orchestrator.challenger.last_recommendation = "revise"
        orchestrator.challenger.last_issues = []

        # After 3 calls total (1 initial + 2 auto-retries), stop
        type(orchestrator.challenger).last_recommendation = PropertyMock(
            side_effect=["revise", "revise", "accept"]
        )

        state = PipelineState(
            user_concept="test",
            synthesis_text="Synthesis",
            tracing_results=[tracing_result_depth0],
        )
        result = orchestrator._node_challenge(state)
        assert orchestrator.challenger.run.call_count == 3


class TestNodePersist:
    """Test persistence node."""

    def test_persist_saves_and_sets_done(self, orchestrator):
        orchestrator.persistence.save.return_value = "output/result.json"
        state = PipelineState(user_concept="test")
        result = orchestrator._node_persist(state)
        assert result.stage == PipelineStage.DONE
        assert result.output_path == "output/result.json"


# ── Error handling ──────────────────────────────────────────────────────────

class TestErrorHandling:
    """Test that exceptions in nodes don't hang the pipeline."""

    def test_exception_in_extractor_propagates(self, orchestrator, seed_paper):
        """An exception in trace_citation raises (LangGraph handles it)."""
        orchestrator.extractor.run.side_effect = RuntimeError("API down")
        state = PipelineState(
            user_concept="test",
            selected_seed=seed_paper,
            current_depth=0,
        )
        with pytest.raises(RuntimeError, match="API down"):
            orchestrator._node_trace_citation(state)


# ── Wrap node adapter ───────────────────────────────────────────────────────

class TestWrapNode:
    """Test the _wrap_node state serialization adapter."""

    def test_wrap_node_converts_dict_to_typed_state(self, orchestrator):
        """_wrap_node converts dict→PipelineState and back."""
        def dummy_node(state: PipelineState) -> PipelineState:
            state.user_concept = "modified"
            return state

        wrapped = orchestrator._wrap_node(dummy_node)
        state_dict = PipelineState(user_concept="original").model_dump()
        result_dict = wrapped(state_dict)
        assert result_dict["user_concept"] == "modified"


# ── Run method ──────────────────────────────────────────────────────────────

class TestOrchestratorRun:
    """Test the run() entry point."""

    def test_run_creates_initial_state_concept_mode(self, orchestrator):
        """run() in concept mode sets up initial state correctly."""
        # Mock the graph invoke to return a done state
        done_state = PipelineState(
            user_concept="attention mechanism",
            stage=PipelineStage.DONE,
        )
        orchestrator.graph = MagicMock()
        orchestrator.graph.invoke.return_value = done_state.model_dump()

        result = orchestrator.run("attention mechanism")
        assert result.stage == PipelineStage.DONE
        call_args = orchestrator.graph.invoke.call_args[0][0]
        assert call_args["user_concept"] == "attention mechanism"
        assert call_args["doi_mode"] is False

    def test_run_creates_initial_state_doi_mode(self, orchestrator):
        """run() in DOI mode sets doi_mode=True."""
        done_state = PipelineState(stage=PipelineStage.DONE, doi_mode=True)
        orchestrator.graph = MagicMock()
        orchestrator.graph.invoke.return_value = done_state.model_dump()

        result = orchestrator.run("", input_doi="10.1234/test")
        call_args = orchestrator.graph.invoke.call_args[0][0]
        assert call_args["doi_mode"] is True
        assert call_args["input_doi"] == "10.1234/test"
