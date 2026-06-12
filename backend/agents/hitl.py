"""
agents/hitl.py - Human-in-the-Loop (HITL) gate implementations.

WHY THREE MANDATORY GATES:
"Unsupervised pipelines drift from user intent, and static models degrade over
time (Mosqueira-Rey et al., 2023). Three mandatory HITL gates ensure the user
can explicitly Approve, Reject, or Refine outputs." (Design proposal §6)

The three gates correspond to the three highest-risk decision points in the pipeline:
  Gate 1 - Seed selection: the seed paper determines the entire citation chain.
            A wrong seed propagates errors through every subsequent agent.
  Gate 2 - Synthesis approval: the synthesis may not accurately represent
            what the user intended to trace. User review before adversarial
            challenge avoids wasting Challenger quota on rejected syntheses.
  Gate 3 - Final output: the last check before data is persisted to the database.
            After this gate, output is written to permanent storage.

IMPLEMENTATION CHOICE - CLI (NOT WEB UI):
The current implementation uses CLI prompts. This is intentional for the MVP:
  1. CLI interaction is testable without a browser or web server
  2. It keeps the HITL mechanism decoupled from the UI layer
  3. The design proposal describes a web interface as a future enhancement;
     the Orchestrator routes to the same HITL gate objects regardless of
     how they present their prompts - swapping CLI for web requires only
     changing this file.

For the demonstration session, CLI HITL gates are actually preferable because
they show the approval flow clearly in the terminal alongside pipeline logs.

References:
  Mosqueira-Rey et al. (2023) Human-in-the-loop machine learning: a state of the art.
    Artificial Intelligence Review, 56(4), 3005-3054.
    doi:10.1007/s10462-022-10246-w
"""

import logging

from core.state import CandidatePaper

logger = logging.getLogger(__name__)


class HITLGate:
    """
    Presents information to the user and waits for explicit approval.

    All gate methods return a boolean (approved) plus any user selection.
    The Orchestrator reads the boolean to determine routing.

    Design decision: gates print to stdout directly rather than using a logging
    framework. This is intentional - HITL prompts are user-facing interaction,
    not diagnostic output. They should always be visible regardless of log level.
    """

    def gate_1_seed_selection(
        self, candidates: list[CandidatePaper]
    ) -> tuple[bool, CandidatePaper | None]:
        """
        Gate 1: present top-N candidate papers and request seed selection.

        Returns (approved=True, selected_paper) if user selects a candidate.
        Returns (approved=False, None) if user rejects all candidates and
        requests a new search (triggers re-retrieval in Orchestrator).
        """
        print("\n" + "=" * 60)
        print("HITL GATE 1 - Seed Paper Selection")
        print("=" * 60)
        print(f"\nFound {len(candidates)} candidate papers for your concept:\n")

        for i, paper in enumerate(candidates, start=1):
            oa_status = "✓ Open Access" if paper.open_access_url else "✗ No OA PDF"
            print(f"  [{i}] {paper.title}")
            print(f"      Authors: {', '.join(paper.authors[:3])}" +
                  (" et al." if len(paper.authors) > 3 else ""))
            print(f"      Year: {paper.year or 'unknown'} | "
                  f"Citations: {paper.citation_count} | {oa_status}")
            if paper.abstract:
                abstract_preview = paper.abstract[:200].replace('\n', ' ')
                print(f"      Abstract: {abstract_preview}...")
            print()

        print("Options:")
        print("  Enter a number (1-{}) to select a seed paper".format(len(candidates)))
        print("  Enter 'r' to reject all and trigger a new search")
        print("  Enter 'q' to quit the pipeline\n")

        while True:
            try:
                raw = input("Your choice: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Non-interactive mode (e.g. in tests) - auto-select first candidate
                logger.warning("HITL Gate 1: non-interactive mode - auto-selecting candidate[0]")
                if candidates:
                    return True, candidates[0]
                return False, None

            if raw == 'q':
                logger.info("HITL Gate 1: user quit the pipeline")
                raise SystemExit(0)

            if raw == 'r':
                print("\nRejected all candidates. Triggering new search...\n")
                logger.info("HITL Gate 1: user rejected all candidates - re-retrieving")
                return False, None

            try:
                choice = int(raw)
                if 1 <= choice <= len(candidates):
                    selected = candidates[choice - 1]
                    print(f"\n✓ Selected: '{selected.title}'")
                    if not selected.open_access_url:
                        print(
                            "\n  ⚠  WARNING: This paper has no open-access PDF.\n"
                            "     Claim extraction will fall back to abstract only.\n"
                            "     Results may be limited - consider selecting an OA paper if available.\n"
                        )
                    else:
                        print()
                    logger.info("HITL Gate 1: user selected paper '%s'", selected.title)
                    return True, selected
                else:
                    print(f"Please enter a number between 1 and {len(candidates)}")
            except ValueError:
                print("Invalid input. Enter a number, 'r', or 'q'.")

    def gate_2_synthesis_approval(self, synthesis_text: str | None) -> bool:
        """
        Gate 2: present synthesis to user and request approval.

        Returns True if approved, False if rejected (triggers re-synthesis).
        """
        print("\n" + "=" * 60)
        print("HITL GATE 2 - Synthesis Review")
        print("=" * 60)
        print("\nGenerated synthesis:\n")
        print(synthesis_text or "(empty synthesis)")
        print("\n" + "-" * 60)
        print("Options: [a]pprove | [r]eject (triggers re-synthesis) | [q]uit\n")

        return self._prompt_approve_reject("Gate 2 synthesis")

    def gate_graph_review(self) -> bool:
        """
        Graph Gate: present concept genealogy graph for user review.

        Returns True if approved (proceed to synthesis), False if rejected (stop pipeline).
        """
        print("\n" + "=" * 60)
        print("HITL GRAPH GATE - Concept Genealogy Graph Review")
        print("=" * 60)
        print("\nThe concept genealogy graph has been built from tracing results.")
        print("Review the graph in the web interface.")
        print("\n" + "-" * 60)
        print("Options: [a]pprove (proceed to synthesis) | [r]eject (stop pipeline) | [q]uit\n")

        return self._prompt_approve_reject("Graph review")

    def gate_3_final_approval(
        self,
        synthesis_text: str | None,
        challenger_critique: str | None,
    ) -> bool:
        """
        Gate 3: present final output (synthesis + challenger critique) for approval.

        This is the last decision point before data is persisted to the database.
        The user sees both the synthesis and the Challenger's review together
        so they can make a fully informed final decision.
        """
        print("\n" + "=" * 60)
        print("HITL GATE 3 - Final Output Approval")
        print("=" * 60)
        print("\n--- SYNTHESIS ---\n")
        print(synthesis_text or "(no synthesis)")

        if challenger_critique:
            print("\n--- CHALLENGER REVIEW ---\n")
            print(challenger_critique)

        print("\n" + "-" * 60)
        print("This is the final approval step. Approving will persist results to the database.")
        print("Options: [a]pprove | [r]eject (returns to synthesis) | [q]uit\n")

        return self._prompt_approve_reject("Gate 3 final output")

    @staticmethod
    def _prompt_approve_reject(gate_name: str) -> bool:
        """Shared approve/reject prompt used by gates 2 and 3."""
        while True:
            try:
                raw = input("Your choice [a/r/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Non-interactive mode (tests) - auto-approve
                logger.warning("HITL %s: non-interactive mode - auto-approving", gate_name)
                return True

            if raw in ('a', 'approve', 'yes', 'y'):
                logger.info("HITL %s: approved", gate_name)
                return True
            elif raw in ('r', 'reject', 'no', 'n'):
                logger.info("HITL %s: rejected - routing to retry", gate_name)
                return False
            elif raw == 'q':
                logger.info("HITL %s: user quit", gate_name)
                raise SystemExit(0)
            else:
                print("Please enter 'a' (approve), 'r' (reject), or 'q' (quit)")