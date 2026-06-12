"""
core/state.py - Global pipeline state schema.

WHY THIS EXISTS:
The system uses a centralised deliberative architecture where a single Orchestrator
manages global state across all agent activations (Dang et al., 2025). By defining
state as a typed Pydantic model, we get schema validation at definition time and
a single source of truth for what data is in-flight at any pipeline stage.

This prevents the "context-window degradation" anti-pattern: reactive agents receive
only the slice of state they need (passed as arguments), never the full context
window. The Orchestrator owns the full StateGraph state; agents are stateless
functions (Brooks, 1991 - subsumption architecture principle).

References:
  Dang et al. (2025) Multi-Agent Collaboration via Evolving Orchestration. arXiv:2505.19591
  Brooks (1991) Intelligence without representation. Artificial Intelligence, 47(1-3), 139-159.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """Explicit stage enumeration keeps routing logic readable and prevents
    string typos from silently misrouting the graph."""
    INIT = "init"
    RETRIEVING = "retrieving"
    AWAITING_SEED_APPROVAL = "awaiting_seed_approval"
    TRACING = "tracing"
    SYNTHESISING = "synthesising"
    AWAITING_SYNTHESIS_APPROVAL = "awaiting_synthesis_approval"
    CHALLENGING = "challenging"
    AWAITING_FINAL_APPROVAL = "awaiting_final_approval"
    PERSISTING = "persisting"
    DONE = "done"
    FAILED = "failed"


class CandidatePaper(BaseModel):
    """Metadata for a single seed paper candidate returned by the Retriever."""
    paper_id: str
    title: str
    authors: list[str]
    year: Optional[int] = None
    citation_count: int = 0
    sjr_score: Optional[float] = None
    open_access_url: Optional[str] = None
    abstract: Optional[str] = None


class ExtractedClaim(BaseModel):
    """A single claim extracted from a paper by the Citation Extractor.

    WHY claim + source_paper_id + depth:
    The system builds an inverted semantic provenance graph. Tracking depth
    and source at claim level (rather than paper level) allows the Evaluator
    to compute conceptual drift relative to the parent claim, not just the
    root concept. This is the core novelty of the recursive tracing design.
    """
    claim_text: str
    source_paper_id: str
    source_title: str
    depth: int  # 0 = seed paper, 1 = direct reference, etc.
    inline_citations: list[str] = Field(default_factory=list)  # DOIs/IDs cited within this claim
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EvaluatedClaim(BaseModel):
    """A claim that has passed through the Evaluator's filtering and scoring.

    WHY separate from ExtractedClaim:
    The Inspector validates structural correctness at every agent boundary
    (Atıl et al., 2025). Keeping Extracted and Evaluated as distinct types
    makes boundary violations immediately visible - if an EvaluatedClaim
    arrives without a validity_score, Pydantic raises before it reaches storage.

    Reference:
      Atıl et al. (2025) Non-determinism of "deterministic" LLM system settings.
      Proceedings of Eval4NLP, ACL 2025.
    """
    claim_text: str
    source_paper_id: str
    source_title: str
    depth: int
    validity_score: float = Field(ge=0.0, le=1.0)
    relevant_concepts: list[str] = Field(default_factory=list)
    drift_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # drift_score measures semantic distance from parent claim embedding.
    # None at depth=0 (seed paper has no parent to drift from).


class TracingResult(BaseModel):
    """Accumulated output of one recursive tracing pass over a paper."""
    paper_id: str
    paper_title: str
    depth: int
    claims: list[EvaluatedClaim] = Field(default_factory=list)
    references_explored: list[str] = Field(default_factory=list)
    references_discarded: list[str] = Field(default_factory=list)


class PipelineState(BaseModel):
    """
    Single global state object threaded through every LangGraph node.

    WHY a flat model rather than nested sub-states:
    LangGraph's StateGraph requires a reducible state dict. Nested Pydantic
    models serialise to nested dicts which LangGraph can update atomically,
    but merging nested updates requires custom reducers. A flat-ish structure
    with explicit Optional fields keeps the reducer logic in one place
    (the Orchestrator) and makes every field's lifecycle explicit.

    Execution constraints (from design proposal Section 5):
      - max_depth: caps recursive citation traversal to prevent graph explosion
        (Lecy and Beatty, 2012 - constrained snowball sampling)
      - max_candidates: Retriever returns at most 3 papers to HITL gate
      - inspector_retries / synthesiser_retries: bounded retry policy
    """
    # ── Pipeline control ──────────────────────────────────────────────────────
    stage: PipelineStage = PipelineStage.INIT
    user_concept: str = ""
    error_message: Optional[str] = None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    candidate_papers: list[CandidatePaper] = Field(default_factory=list)
    selected_seed: Optional[CandidatePaper] = None
    max_candidates: int = 3  # design constraint: present top-3 to user

    # ── Recursive tracing ─────────────────────────────────────────────────────
    current_depth: int = 0
    max_depth: int = 3          # bounding scheme: depth cap (proposal Section 6)
    visited_paper_ids: set[str] = Field(default_factory=set)
    tracing_results: list[TracingResult] = Field(default_factory=list)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    synthesis_text: Optional[str] = None
    challenger_critique: Optional[str] = None  # populated if Challenger is active

    # ── Retry counters ────────────────────────────────────────────────────────
    inspector_retries: int = 0
    max_inspector_retries: int = 1   # design constraint: max 1 retry for Pydantic errors
    synthesiser_retries: int = 0
    max_synthesiser_retries: int = 3  # design constraint: max 3 for Synthesiser/Challenger

    # ── HITL gate flags ───────────────────────────────────────────────────────
    # Three mandatory gates (design proposal Section 5):
    #   Gate 1: seed selection approval
    #   Gate 2: synthesis approval
    #   Gate 3: final output approval
    hitl_gate_1_approved: bool = False
    hitl_gate_2_approved: bool = False
    hitl_gate_3_approved: bool = False

    # ── Output ────────────────────────────────────────────────────────────────
    output_path: Optional[str] = None  # path to final JSON/Markdown deliverable

    class Config:
        # Allow set type in visited_paper_ids - Pydantic v2 handles this natively,
        # but explicit config keeps compatibility with Pydantic v1 shims.
        arbitrary_types_allowed = True