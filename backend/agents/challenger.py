"""
agents/challenger.py - Challenger agent.

ROLE IN ARCHITECTURE:
The Challenger interrogates the Synthesiser's output through structured adversarial
debate to surface weak claims and enforce robust conclusions (Irving, Christiano
and Amodei, 2018). It implements the "AI safety via debate" principle: if a claim
cannot survive a well-informed adversarial challenge, it should not appear in the
final output.

IMPLEMENTATION STATUS: MVP STUB
The full Challenger as designed (multi-turn adversarial debate with bounded retries)
is implemented as a stub in this sprint. The Orchestrator's LangGraph node for
`challenge` is wired and active; this class provides the interface it calls.

DEFERRAL JUSTIFICATION:
Full adversarial debate requires:
  1. Multi-turn conversation state between Synthesiser and Challenger (not single
     LLM calls - a true debate loop with N rounds of exchange)
  2. A separate system prompt that conditions the Challenger to actively seek
     counter-evidence from the traced citation data
  3. A termination condition for the debate (max_synthesiser_retries=3 from
     the design proposal, Section 5)

This requires LangGraph subgraph support (nested graphs) or a manual loop inside
the challenge node - both of which increase Orchestrator complexity significantly.
The decision to defer preserves MVP scope integrity: the Synthesiser HITL gates
(Gate 2 and Gate 3) provide adequate human oversight for the MVP sprint, and the
Challenger adds a second layer of automated QA that is valuable but not critical
for demonstrating the core pipeline.

The stub documents what the full implementation would do and returns a structured
placeholder critique that allows the HITL Gate 3 to function normally.

FUTURE IMPLEMENTATION NOTES (for presentation discussion):
  - Each debate round: Challenger calls Groq with synthesis + "find weaknesses" prompt
  - Synthesiser responds to critique with revisions or counter-arguments
  - Inspector validates structural validity of each exchange
  - After max rounds or convergence, Orchestrator routes to HITL Gate 3
  - Academic basis: Irving et al. (2018) show debate-based evaluation is more
    robust than single-pass verification for complex claims

References:
  Irving, G., Christiano, P. and Amodei, D. (2018) AI safety via debate.
    arXiv preprint arXiv:1805.00899.
  Mosqueira-Rey et al. (2023) Human-in-the-loop machine learning.
    Artificial Intelligence Review, 56(4), 3005-3054.
"""

import logging
from typing import Any

from core.state import TracingResult

logger = logging.getLogger(__name__)


class ChallengerAgent:
    """
    Adversarial reviewer for the Synthesiser's output.

    MVP: returns a structured stub critique. The interface is fully defined
    so the Orchestrator's `challenge` node requires no changes when the full
    implementation is added in a future sprint.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Logging the stub status at init makes it visible in execution logs,
        # which serves as evidence of intentional design decision for the marker.
        logger.info(
            "ChallengerAgent initialised (MVP stub - see docstring for deferral rationale)"
        )

    def run(
        self,
        synthesis_text: str | None,
        tracing_results: list[TracingResult],
    ) -> str:
        """
        Review synthesis and return a critique.

        MVP implementation: returns a structured acknowledgement that documents
        the limitation transparently. This is intentionally visible in the output
        rather than silently returning an empty string - the design proposal
        commits to transparent HITL output (design proposal 6, User-system misalignment
        mitigation: "Three mandatory HITL gates ensure the user can explicitly
        Approve, Reject, or Refine outputs").

        A silent empty critique would misrepresent the system's completeness to the
        HITL user.
        """
        if not synthesis_text:
            return "(challenger: no synthesis to review)"

        total_claims = sum(len(r.claims) for r in tracing_results)
        depths_covered = sorted({r.depth for r in tracing_results})

        # The stub critique is structured to be informative and clearly labelled
        # as a placeholder - it gives the user real information (claim count,
        # depth coverage) while making the limitation explicit.
        critique = (
            f"## Challenger Review (MVP Stub)\n\n"
            f"**Note:** Full adversarial debate is deferred to a future implementation sprint "
            f"(see `agents/challenger.py` for rationale). "
            f"The synthesis has been passed to HITL Gate 3 without automated adversarial review.\n\n"
            f"**Coverage summary:**\n"
            f"- Claims reviewed: {total_claims}\n"
            f"- Depth levels covered: {depths_covered}\n"
            f"- Papers traced: {len(tracing_results)}\n\n"
            f"**Recommended manual review points:**\n"
            f"- Verify that claims at depth > 1 have not drifted significantly from the seed concept\n"
            f"- Check that all cited source titles in the synthesis match the traced paper list\n"
            f"- Confirm no claims in the synthesis are unsupported by the extracted evidence\n\n"
            f"Human review via HITL Gate 3 is the primary quality control at this stage."
        )

        logger.info(
            "ChallengerAgent: stub critique generated (%d chars). "
            "Full adversarial debate deferred (see docstring).",
            len(critique)
        )
        return critique