"""
core/orchestrator.py - Central deliberative Orchestrator.

WHY LANGGRAPH AND NOT A LINEAR DAG:
LangGraph manages workflows through stateful, directed graphs, enabling the
cyclical and iterative reasoning that linear, DAG-based frameworks cannot support
(Sapkota et al., 2025). This system requires cycles because:
  1. Retrieval may loop if the user rejects a seed (HITL Gate 1 rejection).
  2. Synthesis may loop if the user rejects the summary (HITL Gate 2 rejection).
  3. Recursive citation tracing is inherently a loop over depth levels.

WHY CENTRALISED ORCHESTRATION (NOT PEER-TO-PEER):
Agents do not coordinate directly with each other. All activations and context
sharing are managed by this Orchestrator via the evolving global PipelineState
(Dang et al., 2025). This isolates failures at the agent boundary level and
prevents a single misbehaving agent from corrupting another agent's context window.

Architecture classification: hybrid deliberative-reactive (Stone and Veloso, 2000).
This file implements the deliberative layer (global state, intention persistence,
resource-bounded practical reasoning - Bratman et al., 1988).
The agents in /agents/ implement the reactive layer (stateless, respond to inputs).

References:
  Bratman et al. (1988) Plans and resource-bounded practical reasoning.
    Computational Intelligence, 4(4), 349-355.
  Dang et al. (2025) Multi-Agent Collaboration via Evolving Orchestration. arXiv:2505.19591
  Sapkota et al. (2025) LangChain vs. LangGraph vs. LangSmith. TechRxiv.
  Stone and Veloso (2000) Multiagent systems: a survey. Autonomous Robots, 8(3), 345-383.
"""

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from core.state import PipelineState, PipelineStage
from agents.retriever import RetrieverAgent
from agents.citation_extractor import CitationExtractorAgent
from agents.evaluator import EvaluatorAgent
from agents.inspector import InspectorAgent
from agents.synthesiser import SynthesiserAgent
from agents.challenger import ChallengerAgent
from agents.hitl import HITLGate
from storage.persistence import PersistenceLayer

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Builds and runs the LangGraph StateGraph.

    Design decision: the Orchestrator is a class (not a module-level graph)
    so that agent dependencies (API clients, DB connections) can be injected
    at construction time rather than imported as globals. This makes the system
    testable - tests can inject mock agents without patching module state.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._progress = config.get("_progress_callback", lambda msg, level="INFO": None)
        self._init_shared_groq_client()
        self._build_agents()
        self.graph = self._build_graph()

    def _init_shared_groq_client(self) -> None:
        """Create a single rate-limited Groq client shared by all agents."""
        from core.groq_client import RateLimitedGroqClient
        api_key = self.config.get("groq_api_key", "")
        if api_key:
            self.config["_groq_client"] = RateLimitedGroqClient(
                api_key, progress_callback=self._progress
            )
        else:
            self.config["_groq_client"] = None

    def _build_agents(self) -> None:
        """Instantiate all agents with shared config.

        WHY SHARED CONFIG OBJECT:
        API keys, model names, and depth caps are runtime configuration, not
        constants. Passing config explicitly (rather than reading from env
        inside each agent) makes the dependency graph visible and testable.
        """
        self.retriever = RetrieverAgent(self.config)
        self.extractor = CitationExtractorAgent(self.config)
        self.evaluator = EvaluatorAgent(self.config)
        self.inspector = InspectorAgent(self.config)
        self.synthesiser = SynthesiserAgent(self.config)
        self.challenger = ChallengerAgent(self.config)
        self.hitl = HITLGate()
        self.persistence = PersistenceLayer(self.config)

    def _build_graph(self) -> StateGraph:
        """
        Define the LangGraph StateGraph: nodes + conditional edges.

        Node naming convention: snake_case, matches PipelineStage enum values
        where possible. This makes routing conditions readable as:
            state.stage == PipelineStage.RETRIEVING → go to "retrieve"

        Conditional edge functions are defined as inner closures here rather
        than module-level functions so they close over `self` and can access
        agent instances without global state.
        """
        # LangGraph requires state as a dict-like; we use PipelineState.dict()
        # at entry and reconstruct on exit. See _wrap_node() below.
        graph = StateGraph(dict)

        # ── Nodes ─────────────────────────────────────────────────────────────
        graph.add_node("retrieve", self._wrap_node(self._node_retrieve))
        graph.add_node("hitl_gate_1", self._wrap_node(self._node_hitl_gate_1))
        graph.add_node("trace_citation", self._wrap_node(self._node_trace_citation))
        graph.add_node("hitl_graph_review", self._wrap_node(self._node_hitl_graph_review))
        graph.add_node("synthesise", self._wrap_node(self._node_synthesise))
        graph.add_node("hitl_gate_2", self._wrap_node(self._node_hitl_gate_2))
        graph.add_node("challenge", self._wrap_node(self._node_challenge))
        graph.add_node("hitl_gate_3", self._wrap_node(self._node_hitl_gate_3))
        graph.add_node("persist", self._wrap_node(self._node_persist))

        # ── Entry point ───────────────────────────────────────────────────────
        graph.set_entry_point("retrieve")

        # ── Edges ─────────────────────────────────────────────────────────────
        # After retrieval → HITL Gate 1 (mandatory: user must approve seed)
        graph.add_edge("retrieve", "hitl_gate_1")

        # HITL Gate 1: approved → start tracing; rejected → re-retrieve
        graph.add_conditional_edges(
            "hitl_gate_1",
            lambda state: "trace_citation" if state.get("hitl_gate_1_approved") else "retrieve",
        )

        # Tracing loop: recurse until depth cap or no references left
        graph.add_conditional_edges(
            "trace_citation",
            self._route_after_tracing,
        )

        # HITL Graph Review: approved → synthesise; rejected → fail
        graph.add_conditional_edges(
            "hitl_graph_review",
            lambda state: "synthesise" if state.get("hitl_graph_approved") else END,
        )

        # After tracing completes → synthesise
        graph.add_edge("synthesise", "hitl_gate_2")

        # HITL Gate 2: approved → challenge; rejected → re-synthesise
        graph.add_conditional_edges(
            "hitl_gate_2",
            lambda state: "challenge" if state.get("hitl_gate_2_approved") else "synthesise",
        )

        # Challenge → HITL Gate 3
        graph.add_edge("challenge", "hitl_gate_3")

        # HITL Gate 3: approved → persist; rejected → re-synthesise (reset loop)
        graph.add_conditional_edges(
            "hitl_gate_3",
            lambda state: "persist" if state.get("hitl_gate_3_approved") else "synthesise",
        )

        # Persist → done
        graph.add_edge("persist", END)

        return graph.compile()

    def _route_after_tracing(self, state: dict) -> str:
        """
        Routing logic for the breadth-first citation tracing loop.

        BFS strategy: at each depth level, trace up to `papers_per_level` papers.
        When all papers at the current level are done, advance to the next depth
        by collecting references from all papers just traced at this level.

        Three-constraint bounding scheme (design proposal Section 6):
          1. Depth cap: current_depth > max_depth → stop
          2. No pending papers at current level and no new refs for next level → stop
          3. All references already visited → stop (prevents cycles in citation graph)

        Reference:
          Lecy and Beatty (2012) Representative literature reviews using
          constrained snowball sampling. SSRN Electronic Journal.
        """
        current_depth = state.get("current_depth", 0)
        max_depth = state.get("max_depth", 3)
        papers_per_level = state.get("papers_per_level", 3)
        visited = set(state.get("visited_paper_ids", []))
        pending = state.get("pending_papers_at_depth", [])
        tracing_results = state.get("tracing_results", [])

        # If there are still papers queued at the current depth, keep tracing
        if pending:
            return "trace_citation"

        # Current depth level is exhausted — try to advance to next depth
        # Collect all unexplored references from papers traced at this depth level
        next_depth = current_depth
        all_refs_at_current_depth = []
        for result in tracing_results:
            if result.get("depth") == current_depth - 1:
                # References discovered by papers at (current_depth - 1) form
                # the pool for current_depth. But if current_depth was just
                # completed, we need refs from papers at current_depth - 1.
                pass

        # Actually: papers traced at (current_depth - 1) have references that
        # should be traced at current_depth. After the seed (depth 0) is traced,
        # current_depth becomes 1. The seed's references form the depth-1 pool.
        # After depth-1 papers are traced, their references form the depth-2 pool.

        # Collect refs from papers traced at the most recent depth level
        latest_depth = current_depth - 1  # last completed level
        all_new_refs = []
        for result in tracing_results:
            if result.get("depth") == latest_depth:
                for ref_id in result.get("references_explored", []):
                    if ref_id not in visited and ref_id not in all_new_refs:
                        all_new_refs.append(ref_id)

        # Hard failure check: if we just did the seed and found nothing
        if current_depth <= 1 and not all_new_refs:
            logger.error(
                "No references found after tracing seed paper. "
                "Cannot build concept genealogy without at least depth-1 references."
            )
            self._progress(
                "No references found for the seed paper. "
                "Try a different seed paper or concept.",
                "ERROR"
            )
            state["stage"] = PipelineStage.FAILED.value
            state["error_message"] = (
                "No references found for the seed paper in Semantic Scholar. "
                "Concept genealogy requires at least first-order references. "
                "Try a different seed paper or concept query."
            )
            return END

        # If we've reached the depth cap or no new refs, tracing is complete
        if current_depth > max_depth or not all_new_refs:
            logger.info(
                "Tracing complete. depth=%d/%d, new_refs_available=%d",
                current_depth - 1, max_depth, len(all_new_refs)
            )
            return "hitl_graph_review"

        # Populate the queue for the next depth level (up to papers_per_level)
        state["pending_papers_at_depth"] = all_new_refs[:papers_per_level]
        self._progress(
            f"Advancing to depth {current_depth}: "
            f"{min(len(all_new_refs), papers_per_level)} papers to trace"
        )
        return "trace_citation"

    @staticmethod
    def _wrap_node(fn):
        """
        Adapter between LangGraph's dict-based state and our typed PipelineState.

        WHY THIS WRAPPER:
        LangGraph operates on plain dicts for its StateGraph. PipelineState is a
        Pydantic model. Rather than sprinkle .dict() / PipelineState(**state) calls
        throughout every node, we centralise the conversion here. Nodes receive
        and return typed PipelineState objects; this wrapper handles serialisation.

        This is a pragmatic trade-off: we lose LangGraph's native partial-update
        merging (which works on dicts), but we gain full Pydantic validation on
        every node transition - which is exactly the Inspector's job.
        """
        def wrapped(state: dict) -> dict:
            typed_state = PipelineState(**state)
            result = fn(typed_state)
            return result.dict()
        return wrapped

    # ── Node implementations ──────────────────────────────────────────────────

    def _node_retrieve(self, state: PipelineState) -> PipelineState:
        """Activate Retriever agent to fetch top-3 candidate seed papers."""
        logger.info("Stage: RETRIEVING for concept='%s'", state.user_concept)
        state.stage = PipelineStage.RETRIEVING
        self._progress("Searching for seed papers...")
        candidates = self.retriever.run(state.user_concept)
        # Inspector validates structural output before it enters state
        validated = self.inspector.validate_candidates(candidates, state)
        state.candidate_papers = validated
        self._progress(f"Found {len(validated)} candidate papers. Please select a seed.")
        state.stage = PipelineStage.AWAITING_SEED_APPROVAL
        return state

    def _node_hitl_gate_1(self, state: PipelineState) -> PipelineState:
        """HITL Gate 1: present candidates to user, await seed selection."""
        approved, selected = self.hitl.gate_1_seed_selection(state.candidate_papers)
        state.hitl_gate_1_approved = approved
        if approved:
            state.selected_seed = selected
            state.visited_paper_ids.add(selected.paper_id)
        return state

    def _node_trace_citation(self, state: PipelineState) -> PipelineState:
        """
        One iteration of breadth-first citation tracing.

        BFS approach: at each depth level, trace multiple papers (up to
        papers_per_level). The pending_papers_at_depth queue holds paper IDs
        to process at the current depth. When the queue empties, the routing
        function advances to the next depth level and refills the queue.

        The recursive tracing is implemented as repeated node activations
        (LangGraph loop) rather than a Python recursive function call. This
        externalises the recursion state into PipelineState, making it:
          - inspectable at every step (HITL can observe progress)
          - resumable if an API call fails
          - bounded by the Orchestrator's routing logic, not the call stack
        """
        logger.info("Stage: TRACING depth=%d", state.current_depth)
        state.stage = PipelineStage.TRACING

        # Determine which paper to trace
        if state.current_depth == 0:
            # First iteration: trace the seed paper
            target_paper = state.selected_seed
        else:
            # Pop next paper from BFS queue
            target_paper = self._pop_next_paper(state)

        if target_paper is None:
            logger.warning("No target paper at depth %d - terminating trace", state.current_depth)
            # Clear the queue to signal completion of this level
            state.pending_papers_at_depth = []
            return state

        # Count articles traced at this depth for progress reporting
        articles_at_depth = sum(
            1 for r in state.tracing_results if r.depth == state.current_depth
        )
        self._progress(
            f"Depth {state.current_depth}, "
            f"Paper {articles_at_depth + 1} — \"{target_paper.title}\""
        )

        # Extractor → Evaluator → Inspector pipeline
        self._progress("Extracting claims from paper...")
        raw_claims = self.extractor.run(target_paper, state.current_depth)
        self._progress(f"Evaluating {len(raw_claims)} extracted claims...")
        evaluated_claims = self.evaluator.run(raw_claims, state)
        validated_result = self.inspector.validate_tracing_result(
            evaluated_claims, target_paper, state.current_depth, state
        )

        # Fetch the paper's references from Semantic Scholar for the next level
        self._progress("Fetching references...")
        ref_ids = self.retriever.get_references(target_paper.paper_id)
        validated_result.references_explored = ref_ids
        self._progress(
            f"Depth {state.current_depth} — "
            f"{len(validated_result.claims)} claims, {len(ref_ids)} references found"
        )
        logger.info(
            "Depth %d complete: %d claims extracted, %d references found",
            state.current_depth, len(validated_result.claims), len(ref_ids)
        )

        state.tracing_results.append(validated_result)
        state.visited_paper_ids.add(target_paper.paper_id)

        # If this was depth 0 (seed), advance depth counter immediately
        # For deeper levels, depth advances when the queue is exhausted (in routing)
        if state.current_depth == 0:
            state.current_depth = 1
        else:
            # Remove processed paper from the pending queue
            if target_paper.paper_id in state.pending_papers_at_depth:
                state.pending_papers_at_depth.remove(target_paper.paper_id)
            # If queue is now empty, advance depth for next level
            if not state.pending_papers_at_depth:
                state.current_depth += 1

        return state

    def _node_hitl_graph_review(self, state: PipelineState) -> PipelineState:
        """HITL Graph Gate: present concept genealogy graph for approval before synthesis."""
        state.stage = PipelineStage.AWAITING_GRAPH_APPROVAL

        # Notify API layer that tracing is complete so graph data is available
        tracing_cb = self.config.get("_tracing_complete_callback")
        if tracing_cb:
            tracing_cb(state.tracing_results, state.selected_seed)

        self._progress("Tracing complete. Review the genealogy graph.")
        approved = self.hitl.gate_graph_review()
        state.hitl_graph_approved = approved
        if not approved:
            self._progress("Graph rejected. Pipeline stopped.", "WARNING")
        return state

    def _node_synthesise(self, state: PipelineState) -> PipelineState:
        """Activate Synthesiser to produce grounded summary from all traced claims."""
        logger.info("Stage: SYNTHESISING across %d tracing results", len(state.tracing_results))
        state.stage = PipelineStage.SYNTHESISING

        total_claims = sum(len(r.claims) for r in state.tracing_results)
        self._progress(
            f"Synthesising genealogy from {total_claims} claims "
            f"across {len(state.tracing_results)} papers..."
        )
        synthesis = self.synthesiser.run(state.tracing_results, state.user_concept)
        validated = self.inspector.validate_synthesis(synthesis, state)
        state.synthesis_text = validated
        self._progress("Synthesis ready for review.")
        state.stage = PipelineStage.AWAITING_SYNTHESIS_APPROVAL
        return state

    def _node_hitl_gate_2(self, state: PipelineState) -> PipelineState:
        """HITL Gate 2: present synthesis to user, await approval."""
        approved = self.hitl.gate_2_synthesis_approval(state.synthesis_text)
        state.hitl_gate_2_approved = approved
        if not approved:
            state.synthesiser_retries += 1
        return state

    def _node_challenge(self, state: PipelineState) -> PipelineState:
        """Activate Challenger to interrogate synthesis (adversarial review)."""
        state.stage = PipelineStage.CHALLENGING
        self._progress("Running adversarial review on synthesis...")
        critique = self.challenger.run(state.synthesis_text, state.tracing_results)
        state.challenger_critique = critique
        self._progress("Challenge complete. Final review ready.")
        state.stage = PipelineStage.AWAITING_FINAL_APPROVAL
        return state

    def _node_hitl_gate_3(self, state: PipelineState) -> PipelineState:
        """HITL Gate 3: present final output to user, await final approval."""
        approved = self.hitl.gate_3_final_approval(
            state.synthesis_text, state.challenger_critique
        )
        state.hitl_gate_3_approved = approved
        return state

    def _node_persist(self, state: PipelineState) -> PipelineState:
        """Write final output to PostgreSQL + JSON/Markdown files."""
        state.stage = PipelineStage.PERSISTING
        self._progress("Saving results...")
        output_path = self.persistence.save(state)
        state.output_path = output_path
        state.stage = PipelineStage.DONE
        self._progress("Pipeline complete!")
        logger.info("Pipeline complete. Output at: %s", output_path)
        return state

    def _pop_next_paper(self, state: PipelineState) -> "CandidatePaper | None":
        """
        Pop the next paper from the BFS queue and fetch its full metadata.

        WHY SJR RANKING FOR SELECTION:
        SJR (SCImago Journal Rank) is a field-normalised impact metric that accounts
        for citation prestige, not just count (SCImago, n.d.). Using it to prioritise
        which references to trace next implements the 'SJR-ranked limits' component
        of the three-constraint bounding scheme, ensuring we follow the highest-quality
        citation chains first when the depth cap forces us to be selective.
        """
        if not state.pending_papers_at_depth:
            return None

        # Try each queued paper ID until we find one with fetchable metadata
        for ref_id in list(state.pending_papers_at_depth):
            if ref_id in state.visited_paper_ids:
                state.pending_papers_at_depth.remove(ref_id)
                continue
            paper = self.retriever.fetch_paper_by_id(ref_id)
            if paper is not None:
                return paper
            # Couldn't fetch — remove from queue and try next
            state.pending_papers_at_depth.remove(ref_id)

        return None

    def run(self, user_concept: str) -> PipelineState:
        """Entry point: run the full pipeline for a user concept."""
        initial_state = PipelineState(
            user_concept=user_concept,
            max_depth=self.config.get("max_depth", 3),
            papers_per_level=self.config.get("papers_per_level", 3),
        )
        # BFS tracing with fan-out needs more iterations than default 25:
        # worst case = 1 seed + (max_depth * papers_per_level) traces + gates + synthesis
        max_iterations = 10 + (initial_state.max_depth * initial_state.papers_per_level * 2)
        final_state_dict = self.graph.invoke(
            initial_state.dict(),
            config={"recursion_limit": max_iterations},
        )
        return PipelineState(**final_state_dict)