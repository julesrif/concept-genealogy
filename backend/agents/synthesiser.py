"""
agents/synthesiser.py - Synthesiser agent.

ROLE IN ARCHITECTURE:
Reactive specialist agent. Receives validated claims and citation data from all
tracing levels and produces a grounded synthesis - a coherent narrative of the
concept's genealogy across the explored citation chain.

WHY LLAMA-3.3-70B FOR SYNTHESIS (NOT 8B):
Synthesis requires comparative reasoning across multiple papers at multiple
depth levels - comparing how a concept was articulated in the seed paper vs.
its foundational sources, identifying where definitions shifted, and producing
a coherent narrative. This is qualitatively different from the extraction and
filtering tasks (which are classification/scoring tasks well within 8b capacity).

The design proposal §3 specifies: "larger models only where comparative reasoning
is required." Synthesis is explicitly that case.

GROUNDING STRATEGY:
Every claim in the synthesis is attributed to its source paper. The Synthesiser
prompt instructs the model to cite source_paper_id and depth for each claim used.
This implements the "traceable outputs" requirement from the design proposal §1.

HALLUCINATION MITIGATION:
The Synthesiser only has access to the evaluated claims (not the raw paper text).
This is a deliberate constraint: the Evaluator has already filtered for relevance
and validity, so the Synthesiser is working from a curated, high-confidence set.
Synthesising from a curated set (rather than from raw text) reduces the surface
area for hallucination generation (Maynez et al., 2020).

The Challenger (agents/challenger.py) provides adversarial review of this output
as an additional layer of semantic quality assurance (Irving et al., 2018).

References:
  Irving, Christiano and Amodei (2018) AI safety via debate. arXiv:1805.00899.
  Maynez et al. (2020) On faithfulness and factuality. ACL 2020, pp. 1906-1919.
"""

import logging
from typing import Any

from core.state import TracingResult

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT = """You are a scientific synthesis writer. Your task is to write a grounded genealogy
of the concept "{concept}" based EXCLUSIVELY on the extracted claims provided below.

STRICT RULES:
- Use ONLY the claims provided. Do NOT use your general knowledge about the concept.
- If a claim is used, cite its source in the format [Source: <paper_title>, depth=<depth>].
- If the claims are insufficient to write a full genealogy, say so explicitly and summarise
  only what the available claims actually support. Do NOT fill gaps with general knowledge.
- Never write phrases like "typically", "generally understood as", or "would expect to find"
  - these signal you are drawing on parametric knowledge rather than the provided claims.

The claims below are organised by depth (0 = seed paper, 1 = direct references, 2+ = deeper sources).

Write a coherent 4-6 paragraph academic synthesis that:
1. Introduces the concept as defined in the seed paper claims (depth=0)
2. Traces how the concept evolved in cited works (depth=1+)
3. Identifies any conceptual drift visible in the claims
4. Concludes with a summary grounded only in the provided evidence

Extracted claims (organised by depth):
{claims_by_depth}

Synthesis (grounded claims only - no general knowledge):"""


class SynthesiserAgent:
    """
    Produces grounded academic synthesis from evaluated claims across all depth levels.

    Design decision: the Synthesiser receives TracingResult objects (which contain
    EvaluatedClaim lists) rather than raw claim strings. This keeps the synthesis
    grounded to validated, scored claims only - never to raw LLM output from
    the Extractor.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = config.get("groq_model_reasoning", "llama-3.3-70b-versatile")
        self._init_groq_client()

    def _init_groq_client(self) -> None:
        try:
            from groq import Groq
            self.groq = Groq(api_key=self.config.get("groq_api_key", ""))
        except ImportError:
            logger.error("groq package not installed")
            self.groq = None

    def run(
        self,
        tracing_results: list[TracingResult],
        user_concept: str,
    ) -> str:
        """
        Main entry point. Returns Markdown synthesis string.

        Retry logic is handled by the Orchestrator (max_synthesiser_retries=3).
        The Synthesiser itself does not retry - it returns whatever the LLM produces
        and the Inspector/HITL gates decide whether a retry is needed.
        """
        if not tracing_results:
            logger.warning("Synthesiser: no tracing results to synthesise")
            return "(synthesis skipped - no tracing results available)"

        # Count total claims across all results
        total_claims = sum(len(r.claims) for r in tracing_results)
        if total_claims == 0:
            # Hard stop: refuse to synthesise with zero claims.
            # WHY: a Synthesiser with no grounded claims will hallucinate from
            # parametric knowledge (Maynez et al., 2020). Returning an explicit
            # failure message is safer than producing a plausible-but-fabricated
            # synthesis the user might not recognise as ungrounded.
            logger.error(
                "Synthesiser: 0 claims extracted across %d tracing results - "
                "refusing to synthesise. Likely cause: no OA PDF and empty abstract.",
                len(tracing_results)
            )
            return (
                "## Synthesis Failed - No Claims Extracted\n\n"
                "The pipeline could not extract any claims from the selected paper(s).\n\n"
                "**Most likely cause:** the selected seed paper has no open-access PDF "
                "and its abstract was empty or unavailable from Semantic Scholar.\n\n"
                "**Recommended action:** reject this synthesis (press 'r') and select "
                "a different seed paper - preferably one marked ✓ Open Access."
            )

        claims_by_depth = self._format_claims_by_depth(tracing_results)

        if not self.groq:
            logger.error("Synthesiser: Groq client unavailable - returning placeholder")
            return self._placeholder_synthesis(tracing_results, user_concept)

        prompt = SYNTHESIS_PROMPT.format(
            concept=user_concept,
            claims_by_depth=claims_by_depth,
        )

        try:
            response = self.groq.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                # WHY temperature=0.3 (not 0):
                # Unlike extraction (where determinism is critical for validation),
                # synthesis benefits from slight variation in phrasing to produce
                # natural academic prose. 0.3 introduces minimal variance while
                # keeping the output grounded and coherent.
                max_tokens=1500,
            )
            synthesis = response.choices[0].message.content
            logger.info("Synthesiser: generated synthesis (%d chars)", len(synthesis))
            return synthesis

        except Exception as e:
            logger.error("Synthesiser: Groq API call failed: %s", e)
            return f"(synthesis failed: {e})"

    @staticmethod
    def _format_claims_by_depth(tracing_results: list[TracingResult]) -> str:
        """
        Format evaluated claims grouped by depth level for the synthesis prompt.

        WHY GROUP BY DEPTH:
        The synthesis prompt instructs the model to trace how the concept
        evolved across depth levels. Providing claims pre-grouped by depth
        reduces the cognitive load on the LLM and makes it less likely to
        conflate claims from different genealogical levels.
        """
        sections: list[str] = []
        for result in sorted(tracing_results, key=lambda r: r.depth):
            depth_label = "Seed paper" if result.depth == 0 else f"Depth {result.depth}"
            header = f"\n--- {depth_label}: {result.paper_title} ---"
            claim_lines = [
                f"  - [{c.validity_score:.2f}] {c.claim_text}"
                for c in result.claims
                if c.claim_text.strip()
            ]
            if claim_lines:
                sections.append(header + "\n" + "\n".join(claim_lines))

        return "\n".join(sections) if sections else "(no claims)"

    @staticmethod
    def _placeholder_synthesis(
        tracing_results: list[TracingResult],
        user_concept: str,
    ) -> str:
        """
        Fallback synthesis when Groq is unavailable.

        Returns a structured placeholder that documents what was found without
        LLM-generated prose. Used in testing and offline demonstration.
        This is intentionally minimal - the presence of this method documents
        that the system degrades gracefully rather than crashing.
        """
        total_claims = sum(len(r.claims) for r in tracing_results)
        depths = sorted({r.depth for r in tracing_results})
        papers = [r.paper_title for r in tracing_results]

        return (
            f"## Synthesis of '{user_concept}' (placeholder - LLM unavailable)\n\n"
            f"Traced {len(tracing_results)} papers across depths {depths}.\n"
            f"Total evaluated claims: {total_claims}\n\n"
            f"Papers explored:\n" +
            "\n".join(f"  - {p}" for p in papers)
        )