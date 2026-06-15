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
            state.stage == PipelineStage.RETRIEVING -> go to "retrieve"

        Conditional edge functions are defined as inner closures here rather
        than module-level functions so they close over `self` and can access
        agent instances without global state.
        """
        # LangGraph requires state as a dict-like; we use PipelineState.dict()
        # at entry and reconstruct on exit. See _wrap_node() below.
        graph = StateGraph(dict)

        # Nodes
        graph.add_node("validate_doi", self._wrap_node(self._node_validate_doi))
        graph.add_node("extract_concepts", self._wrap_node(self._node_extract_concepts))
        graph.add_node("hitl_concept_selection", self._wrap_node(self._node_hitl_concept_selection))
        graph.add_node("retrieve", self._wrap_node(self._node_retrieve))
        graph.add_node("hitl_gate_1", self._wrap_node(self._node_hitl_gate_1))
        graph.add_node("trace_citation", self._wrap_node(self._node_trace_citation))
        graph.add_node("hitl_graph_review", self._wrap_node(self._node_hitl_graph_review))
        graph.add_node("synthesise", self._wrap_node(self._node_synthesise))
        graph.add_node("hitl_gate_2", self._wrap_node(self._node_hitl_gate_2))
        graph.add_node("challenge", self._wrap_node(self._node_challenge))
        graph.add_node("hitl_gate_3", self._wrap_node(self._node_hitl_gate_3))
        graph.add_node("persist", self._wrap_node(self._node_persist))

        # Entry point: route based on whether DOI mode is active
        graph.set_entry_point("validate_doi")

        # validate_doi routes: DOI mode -> extract_concepts; concept mode -> retrieve
        graph.add_conditional_edges(
            "validate_doi",
            self._route_after_doi_validation,
        )

        # extract_concepts -> concept selection HITL gate
        graph.add_edge("extract_concepts", "hitl_concept_selection")

        # Concept selection gate: approved -> trace_citation; rejected -> END
        graph.add_conditional_edges(
            "hitl_concept_selection",
            lambda state: "trace_citation" if state.get("hitl_gate_1_approved") else END,
        )

        # After retrieval -> HITL Gate 1 (mandatory: user must approve seed)
        graph.add_edge("retrieve", "hitl_gate_1")

        # HITL Gate 1: approved -> start tracing; rejected -> re-retrieve
        graph.add_conditional_edges(
            "hitl_gate_1",
            lambda state: "trace_citation" if state.get("hitl_gate_1_approved") else "retrieve",
        )

        # Tracing loop: recurse until depth cap or no references left
        graph.add_conditional_edges(
            "trace_citation",
            self._route_after_tracing,
        )

        # HITL Graph Review: approved -> synthesise; rejected -> fail
        graph.add_conditional_edges(
            "hitl_graph_review",
            lambda state: "synthesise" if state.get("hitl_graph_approved") else END,
        )

        # After tracing completes -> synthesise
        graph.add_edge("synthesise", "hitl_gate_2")

        # HITL Gate 2: approved -> challenge; rejected -> re-synthesise
        graph.add_conditional_edges(
            "hitl_gate_2",
            lambda state: "challenge" if state.get("hitl_gate_2_approved") else "synthesise",
        )

        # Challenge -> HITL Gate 3
        graph.add_edge("challenge", "hitl_gate_3")

        # HITL Gate 3: approved -> persist; rejected -> re-synthesise (reset loop)
        graph.add_conditional_edges(
            "hitl_gate_3",
            lambda state: "persist" if state.get("hitl_gate_3_approved") else "synthesise",
        )

        # Persist -> done
        graph.add_edge("persist", END)

        return graph.compile()

    def _route_after_doi_validation(self, state: dict) -> str:
        """Route after DOI validation: DOI mode -> extract_concepts; else -> retrieve."""
        if state.get("doi_mode"):
            # In DOI mode: if DOI validation failed, go to END
            if state.get("stage") == PipelineStage.FAILED.value:
                return END
            return "extract_concepts"
        else:
            # Concept mode: skip DOI validation, go to retrieve
            return "retrieve"

    def _route_after_tracing(self, state: dict) -> str:
        """
        Routing logic for the breadth-first citation tracing loop.

        PURE ROUTING FUNCTION — does NOT modify state. All state mutations
        happen inside nodes (where _wrap_node serialises them back to the
        LangGraph state dict). Mutating state here has no effect because
        LangGraph passes a copy to conditional edge functions.

        BFS strategy: at each depth level, trace up to `papers_per_level` papers.
        When all papers at the current level are done, advance to the next depth
        by collecting references from all papers just traced at this level.

        Three-constraint bounding scheme (design proposal Section 6):
          1. Depth cap: current_depth > max_depth -> stop
          2. No pending papers at current level and no new refs for next level -> stop
          3. All references already visited -> stop (prevents cycles in citation graph)

        Reference:
          Lecy and Beatty (2012) Representative literature reviews using
          constrained snowball sampling. SSRN Electronic Journal.
        """
        current_depth = state.get("current_depth", 0)
        max_depth = state.get("max_depth", 3)
        visited = set(state.get("visited_paper_ids", []))
        pending = state.get("pending_papers_at_depth", [])
        tracing_results = state.get("tracing_results", [])

        # If there are still papers queued at the current depth, keep tracing
        if pending:
            return "trace_citation"

        # Current depth level is exhausted — check if next level has refs
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

        # There ARE new refs to trace — route back to trace_citation.
        # The node will populate its own queue (state mutation must happen there).
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

    # Node implementations

    def _node_validate_doi(self, state: PipelineState) -> PipelineState:
        """Validate DOI if in DOI mode; pass-through if in concept mode."""
        if not state.doi_mode:
            # Concept mode: skip validation, routing will send to retrieve
            return state

        state.stage = PipelineStage.VALIDATING_DOI
        self._progress(f"Validating DOI: {state.input_doi}")

        if not state.input_doi:
            state.stage = PipelineStage.FAILED
            state.error_message = "DOI mode requires a DOI to be provided."
            return state

        # Check if DOI exists via CrossRef
        if not self.retriever.validate_doi(state.input_doi):
            state.stage = PipelineStage.FAILED
            state.error_message = f"DOI not found: {state.input_doi}. Please check the DOI and try again."
            self._progress(f"DOI validation failed: {state.input_doi}", "ERROR")
            return state

        # Fetch the paper metadata from Semantic Scholar
        paper = self.retriever.fetch_paper_by_doi(state.input_doi)
        if not paper:
            state.stage = PipelineStage.FAILED
            state.error_message = (
                f"DOI {state.input_doi} exists in CrossRef but could not be found in "
                f"Semantic Scholar. The paper may not be indexed yet."
            )
            self._progress("Paper not found in Semantic Scholar", "ERROR")
            return state

        # In DOI mode, the provided paper IS the Gold Standard (seed paper)
        state.selected_seed = paper
        state.visited_paper_ids.add(paper.paper_id)
        self._progress(f"DOI validated. Paper: \"{paper.title}\"")
        return state

    def _node_extract_concepts(self, state: PipelineState) -> PipelineState:
        """Extract key concepts from the DOI paper for user selection."""
        state.stage = PipelineStage.EXTRACTING_CONCEPTS
        self._progress("Extracting concepts from the paper...")

        paper = state.selected_seed
        if not paper:
            state.stage = PipelineStage.FAILED
            state.error_message = "No seed paper available for concept extraction."
            return state

        # Get paper text (OA or Elsevier or abstract)
        text = ""
        if paper.open_access_url:
            text, _ = self.extractor._extract_from_pdf(paper.open_access_url)
        if not text.strip() and paper.doi and self.extractor.elsevier:
            text = self.extractor.elsevier.get_full_text(paper.doi) or ""
        if not text.strip():
            text = paper.abstract or ""

        if not text.strip():
            state.stage = PipelineStage.FAILED
            state.error_message = "Could not retrieve paper text for concept extraction."
            self._progress("No text available for concept extraction", "ERROR")
            return state

        # Use LLM to extract key concepts
        concepts = self._extract_concepts_from_text(text, paper.title)
        if not concepts:
            state.stage = PipelineStage.FAILED
            state.error_message = "Could not extract concepts from the paper."
            self._progress("Concept extraction failed", "ERROR")
            return state

        state.extracted_concepts = concepts
        state.stage = PipelineStage.AWAITING_CONCEPT_SELECTION
        self._progress(f"Extracted {len(concepts)} concepts. Please select up to 3 to trace.")
        return state

    def _extract_concepts_from_text(self, text: str, title: str) -> list[str]:
        """Use LLM to extract key academic concepts from paper text."""
        groq = self.config.get("_groq_client")
        if not groq:
            return []

        # Truncate text to fit context window
        truncated = text[:8000]
        prompt = (
            f"You are an academic concept analyst. Given the following paper, "
            f"extract the KEY ACADEMIC CONCEPTS that this paper introduces, develops, "
            f"or significantly contributes to.\n\n"
            f"Paper title: \"{title}\"\n"
            f"Paper text:\n{truncated}\n\n"
            f"Return ONLY a JSON array of concept strings (5-15 concepts). "
            f"Each concept should be 1-5 words, representing a distinct academic idea "
            f"that could be traced through citation chains.\n"
            f"Focus on: novel methods, frameworks, mechanisms, theories, and key techniques.\n"
            f"Do NOT include generic terms like 'machine learning' or 'deep learning' unless "
            f"the paper specifically advances them.\n\n"
            f"JSON array:"
        )

        try:
            response = groq.chat_completion(
                model=self.config.get("groq_model_fast", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            # Parse JSON array
            import json
            # Handle potential code fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            concepts = json.loads(raw)
            if isinstance(concepts, list):
                return [str(c).strip() for c in concepts if c and str(c).strip()][:15]
        except Exception as e:
            logger.warning("Concept extraction LLM call failed: %s", e)
        return []

    def _node_hitl_concept_selection(self, state: PipelineState) -> PipelineState:
        """HITL gate: present extracted concepts for user to select up to 3."""
        selected = self.hitl.gate_concept_selection(state.extracted_concepts)
        if selected:
            state.selected_concepts = selected[:3]
            # Use the first selected concept as the primary tracing concept
            state.user_concept = state.selected_concepts[0]
            state.hitl_gate_1_approved = True
            self._progress(
                f"Concepts selected: {', '.join(state.selected_concepts)}. "
                f"Primary concept for tracing: \"{state.user_concept}\""
            )
        else:
            state.hitl_gate_1_approved = False
            self._progress("Concept selection rejected. Pipeline stopped.", "WARNING")
        return state

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
            # If the queue is empty, populate it from the previous depth's refs.
            # This MUST happen here (in a node) not in the routing function,
            # because LangGraph routing functions receive a state copy — mutations
            # there are not persisted back to the graph state.
            if not state.pending_papers_at_depth:
                self._populate_next_depth_queue(state)

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

        # Extractor -> Evaluator -> Inspector pipeline
        self._progress("Extracting claims from paper...")
        raw_claims = self.extractor.run(target_paper, state.current_depth)
        self._progress(f"Evaluating {len(raw_claims)} extracted claims...")
        evaluated_claims = self.evaluator.run(raw_claims, state)
        validated_result = self.inspector.validate_tracing_result(
            evaluated_claims, target_paper, state.current_depth, state
        )

        # Fetch the paper's references from Semantic Scholar for the next level
        self._progress("Fetching references...")
        # Ensure DOI is available for CrossRef lookup — S2 search may not always
        # return externalIds, but individual paper fetch does.
        doi = target_paper.doi
        if not doi and target_paper.paper_id:
            logger.info("DOI missing on paper %s — fetching from S2 detail endpoint", target_paper.paper_id)
            detail = self.retriever.fetch_paper_by_id(target_paper.paper_id)
            if detail and detail.doi:
                doi = detail.doi
                target_paper.doi = doi
                logger.info("Resolved DOI: %s", doi)
        ref_ids = self.retriever.get_references(target_paper.paper_id, doi=doi)
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
            # Set the initial concept at depth 0
            state.concept_at_depth[0] = state.user_concept
            state.current_depth = 1
            # Compute concept abstraction for the next depth
            self._abstract_concept_for_depth(state, completed_depth=0)
        else:
            # Remove processed paper from the pending queue
            if target_paper.paper_id in state.pending_papers_at_depth:
                state.pending_papers_at_depth.remove(target_paper.paper_id)
            # If queue is now empty, advance depth for next level
            if not state.pending_papers_at_depth:
                # Compute concept abstraction before advancing
                self._abstract_concept_for_depth(state, completed_depth=state.current_depth)
                state.current_depth += 1

        return state

    def _node_hitl_graph_review(self, state: PipelineState) -> PipelineState:
        """HITL Graph Gate: present concept genealogy graph for approval before synthesis."""
        state.stage = PipelineStage.AWAITING_GRAPH_APPROVAL

        # Notify API layer that tracing is complete so graph data is available
        tracing_cb = self.config.get("_tracing_complete_callback")
        if tracing_cb:
            tracing_cb(state.tracing_results, state.selected_seed, state.concept_at_depth)

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
        critique = self.challenger.run(
            state.synthesis_text,
            state.tracing_results,
            user_concept=state.user_concept,
        )
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
    def _populate_next_depth_queue(self, state: "PipelineState") -> None:
        """
        Populate pending_papers_at_depth from the previous depth's references.

        Called at the start of trace_citation (depth > 0) when the queue is empty.
        Collects all unvisited reference IDs from papers traced at (current_depth - 1),
        fetches their metadata, ranks by SJR, and takes the top papers_per_level.
        This ensures we trace the highest-quality references first.
        """
        latest_depth = state.current_depth - 1
        papers_per_level = state.papers_per_level or 3
        all_new_refs = []
        for result in state.tracing_results:
            if result.depth == latest_depth:
                for ref_id in result.references_explored:
                    if ref_id not in state.visited_paper_ids and ref_id not in all_new_refs:
                        all_new_refs.append(ref_id)

        if not all_new_refs:
            state.pending_papers_at_depth = []
            return

        # Fetch metadata for candidates and rank by SJR before selecting top-N.
        # We fetch up to 2x papers_per_level to have a good pool for ranking,
        # while not wasting too many API calls on papers we won't trace.
        fetch_pool_size = min(len(all_new_refs), papers_per_level * 2)
        candidates = []
        for ref_id in all_new_refs[:fetch_pool_size]:
            paper = self.retriever.fetch_paper_by_id(ref_id)
            if paper:
                candidates.append(paper)

        if candidates:
            # Rank by SJR (same logic as retriever's _rank_by_sjr)
            ranked = self.retriever._rank_by_sjr(candidates)
            state.pending_papers_at_depth = [p.paper_id for p in ranked[:papers_per_level]]
            # Cache the fetched metadata so _pop_next_paper doesn't re-fetch
            if not hasattr(state, '_paper_cache'):
                state._paper_cache = {}
            for p in ranked[:papers_per_level]:
                state._paper_cache[p.paper_id] = p
        else:
            state.pending_papers_at_depth = all_new_refs[:papers_per_level]

        self._progress(
            f"Advancing to depth {state.current_depth}: "
            f"{len(state.pending_papers_at_depth)} papers to trace"
        )
        logger.info(
            "Populated depth %d queue: %d/%d new refs (papers_per_level=%d, ranked by SJR)",
            state.current_depth, len(state.pending_papers_at_depth),
            len(all_new_refs), papers_per_level
        )

    def _abstract_concept_for_depth(self, state: PipelineState, completed_depth: int) -> None:
        """
        Identify the precursor/ancestor concept for the next depth level.

        WHY CONCEPT ABSTRACTION:
        As citation tracing moves backward in time, modern terminology often
        did not exist in older papers. "Attention mechanism" (Vaswani et al., 2017)
        traces back to "selective attention" (Treisman, 1964) or "saliency maps"
        (Koch and Ullman, 1985). Without abstracting the concept, the relevance
        filter at deeper levels would discard papers that ARE ancestral but use
        different terminology. This step asks the LLM to identify what the concept
        was called (or what broader idea it belonged to) in the era of the papers
        being traced, enabling the evaluator to match against the abstracted term.

        The abstraction may also determine that the concept originated at the
        current depth (not deeper), which is stored for the synthesis narrative.
        """
        groq = self.config.get("_groq_client")
        if not groq:
            # Fallback: use original concept at all depths
            state.concept_at_depth[completed_depth + 1] = state.user_concept
            return

        # Gather claims from the completed depth
        depth_claims = []
        for result in state.tracing_results:
            if result.depth == completed_depth:
                for claim in result.claims:
                    depth_claims.append(claim.claim_text)

        if not depth_claims:
            # No claims extracted — carry forward the previous concept
            prev_concept = state.concept_at_depth.get(completed_depth, state.user_concept)
            state.concept_at_depth[completed_depth + 1] = prev_concept
            return

        claims_text = "\n".join(f"- {c}" for c in depth_claims[:15])  # cap to avoid token overflow
        current_concept = state.concept_at_depth.get(completed_depth, state.user_concept)

        prompt = (
            f"You are tracing the genealogy of the academic concept: \"{state.user_concept}\"\n"
            f"At the current citation depth, the concept is understood as: \"{current_concept}\"\n\n"
            f"Below are claims extracted from papers at this depth level:\n{claims_text}\n\n"
            f"Based on these claims and the citation chain, what is the PRECURSOR or ANCESTOR "
            f"concept that these papers are building upon? This is the broader or older concept "
            f"that existed BEFORE \"{current_concept}\" was formalised — the idea it evolved from.\n\n"
            f"Rules:\n"
            f"- If the concept appears to have ORIGINATED at this depth (i.e., the papers introduce "
            f"it as novel rather than building on a prior idea), respond with the same concept name.\n"
            f"- Otherwise, identify the precursor terminology used in the referenced older literature.\n"
            f"- Keep it concise: 1-5 words, an academic concept name.\n\n"
            f"Respond with ONLY the concept name, no explanation:"
        )

        try:
            response = groq.chat_completion(
                model=self.config.get("groq_model_fast", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            abstracted = response.choices[0].message.content.strip().strip('"\'')
            state.concept_at_depth[completed_depth + 1] = abstracted
            self._progress(
                f"Concept abstraction: depth {completed_depth} \"{current_concept}\" "
                f"→ depth {completed_depth + 1} \"{abstracted}\""
            )
            logger.info(
                "Concept abstraction: depth %d '%s' -> depth %d '%s'",
                completed_depth, current_concept, completed_depth + 1, abstracted
            )
        except Exception as e:
            logger.warning("Concept abstraction failed: %s — using previous concept", e)
            state.concept_at_depth[completed_depth + 1] = current_concept

    def _pop_next_paper(self, state: PipelineState) -> "CandidatePaper | None":
        """
        Pop the next paper from the BFS queue and return its metadata.

        Uses the paper cache populated by _populate_next_depth_queue to avoid
        redundant S2 API calls. Falls back to fetching if not cached.
        """
        if not state.pending_papers_at_depth:
            return None

        cache = getattr(state, '_paper_cache', {})

        # Try each queued paper ID until we find one with fetchable metadata
        for ref_id in list(state.pending_papers_at_depth):
            if ref_id in state.visited_paper_ids:
                state.pending_papers_at_depth.remove(ref_id)
                continue
            # Check cache first (populated during SJR ranking in _populate_next_depth_queue)
            if ref_id in cache:
                return cache.pop(ref_id)
            paper = self.retriever.fetch_paper_by_id(ref_id)
            if paper is not None:
                return paper
            # Couldn't fetch — remove from queue and try next
            state.pending_papers_at_depth.remove(ref_id)

        return None

    def run(self, user_concept: str, input_doi: str | None = None) -> PipelineState:
        """Entry point: run the full pipeline for a user concept or DOI."""
        doi_mode = bool(input_doi and input_doi.strip())
        initial_state = PipelineState(
            user_concept=user_concept if not doi_mode else "",
            input_doi=input_doi.strip() if doi_mode else None,
            doi_mode=doi_mode,
            max_depth=self.config.get("max_depth", 3),
            papers_per_level=self.config.get("papers_per_level", 3),
        )
        # BFS tracing with fan-out needs more iterations than LangGraph default (25):
        # Each depth level traces papers_per_level papers (one node invocation each),
        # plus HITL gates, synthesis, challenge, and persistence nodes.
        # Formula: seed(1) + depth*papers_per_level + gates(4) + synth+challenge+persist(3) + margin
        max_iterations = 20 + (initial_state.max_depth * initial_state.papers_per_level * 4)
        final_state_dict = self.graph.invoke(
            initial_state.dict(),
            config={"recursion_limit": max_iterations},
        )
        return PipelineState(**final_state_dict)