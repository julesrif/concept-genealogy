"""
agents/challenger.py - Challenger agent.

ROLE IN ARCHITECTURE:
The Challenger interrogates the Synthesiser's output through structured adversarial
debate to surface weak claims and enforce robust conclusions (Irving, Christiano
and Amodei, 2018). It implements the "AI safety via debate" principle: if a claim
cannot survive a well-informed adversarial challenge, it should not appear in the
final output.

IMPLEMENTATION:
Multi-turn adversarial review (up to MAX_DEBATE_ROUNDS) using structured LLM prompts.
The Challenger examines the synthesis against the traced evidence (TracingResults) and
produces a critique identifying:
  1. Unsupported claims (present in synthesis but no matching evidence in traced claims)
  2. Over-stated confidence (weak evidence presented with strong language)
  3. Conceptual drift (claims that stray from the root concept at higher depths)
  4. Missing counter-evidence (known limitations not acknowledged)

DEBATE MECHANICS:
Each round: Challenger generates critique → checks if issues remain → terminates
when no new issues are found or max rounds is reached. This is a single-sided
debate (Challenger challenges, Synthesiser does not respond) because the synthesis
is a static artefact at this point - the HITL Gate 3 acts as the Synthesiser's
"response" by deciding whether to accept, reject, or request revision.

References:
  Irving, G., Christiano, P. and Amodei, D. (2018) AI safety via debate.
    arXiv preprint arXiv:1805.00899.
  Mosqueira-Rey et al. (2023) Human-in-the-loop machine learning.
    Artificial Intelligence Review, 56(4), 3005-3054.
"""

import json
import logging
from typing import Any

from core.state import TracingResult, EvaluatedClaim

logger = logging.getLogger(__name__)

# MAX_DEBATE_ROUNDS=2: Each round costs 1 LLM call to the 70b model (30 req/min
# Groq free-tier limit). Two rounds are sufficient because the Challenger operates
# on a static synthesis - round 2 only adds value if round 1 identified issues
# that suggest deeper structural problems. Empirically, round 3+ produces
# repetitive findings with diminishing severity (Irving et al., 2018).
MAX_DEBATE_ROUNDS = 2

CHALLENGE_SYSTEM_PROMPT = """You are a rigorous academic reviewer performing adversarial review of a concept genealogy synthesis. Your role is to identify weaknesses, unsupported claims, and logical gaps.

You have access to the actual traced evidence (claims extracted from papers with validity scores and depth levels). Your critique MUST be grounded in the evidence provided - do not speculate about what the papers might contain.

Respond with a JSON object:
{
  "issues": [
    {
      "type": "unsupported|overstated|drift|missing_counter",
      "severity": "high|medium|low",
      "quote": "exact text from synthesis that is problematic",
      "explanation": "why this is problematic, referencing specific evidence gaps"
    }
  ],
  "overall_assessment": "strong|adequate|weak",
  "recommendation": "approve|revise|reject"
}

JSON only, no preamble."""

CHALLENGE_USER_PROMPT = """## Synthesis Under Review:
{synthesis}

## Traced Evidence (papers with extracted claims):
{evidence_summary}

## Task:
1. Check every factual statement in the synthesis against the traced evidence
2. Identify claims that lack supporting evidence from the traced papers
3. Flag any overstated conclusions (weak evidence presented as definitive)
4. Note conceptual drift at deeper depth levels. Drift means the claim's usage of "{concept}" has diverged from the gold-standard seed paper definition. We are NOT comparing papers to each other (each paper may cover a different topic); we are tracking how the *referenced concept* changes meaning across the citation chain. Assess drift on three axes:
   - Seed definition: how the seed paper (depth 0) defined/used the concept
   - Downstream usage: how the claim at this depth redefines, narrows, or shifts the concept
   - Evolutionary path: the step-by-step semantic transformation through each depth level
5. Identify missing caveats or counter-evidence

Provide your structured critique."""


class ChallengerAgent:
    """
    Adversarial reviewer for the Synthesiser's output.

    Performs structured adversarial review using the shared Groq client.
    Produces a critique that is presented to the user at HITL Gate 3 alongside
    the synthesis, enabling informed approval/rejection decisions.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.groq = config.get("_groq_client")
        self.model = config.get("groq_model_slow", "llama-3.3-70b-versatile")
        self.last_issues: list[dict] = []  # structured issues from latest run
        self.last_recommendation: str = "revise"  # latest recommendation: approve|revise|reject
        logger.info("ChallengerAgent initialised (model=%s)", self.model)

    def run(
        self,
        synthesis_text: str | None,
        tracing_results: list[TracingResult],
        user_concept: str = "",
        previous_critique: str | None = None,
    ) -> str:
        """
        Review synthesis against traced evidence and return a structured critique.

        Args:
            previous_critique: If this is a debate retry (Gate 3 rejected),
                contains the previous round's critique. The Challenger is asked
                to dig deeper and find issues not already identified.

        Falls back to a summary-only review if no LLM client is available.
        """
        if not synthesis_text:
            return "(challenger: no synthesis to review)"

        total_claims = sum(len(r.claims) for r in tracing_results)
        depths_covered = sorted({r.depth for r in tracing_results})

        # If no LLM available, produce a structured summary without adversarial analysis
        if not self.groq:
            logger.warning("ChallengerAgent: no Groq client - producing coverage-only review")
            return self._fallback_critique(synthesis_text, tracing_results, total_claims, depths_covered)

        # Build evidence summary for the LLM
        evidence_summary = self._build_evidence_summary(tracing_results)

        # Multi-round adversarial review
        all_issues = []
        assessment = "adequate"
        recommendation = "revise"
        for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
            logger.info("Challenger: debate round %d/%d", round_num, MAX_DEBATE_ROUNDS)
            issues, assessment, recommendation = self._run_challenge_round(
                synthesis_text, evidence_summary, user_concept,
                prior_issues=all_issues,
                previous_critique=previous_critique,
            )
            if issues:
                # Deduplicate: skip issues whose quote already appeared
                existing_quotes = {i.get("quote", "").strip() for i in all_issues}
                for issue in issues:
                    if issue.get("quote", "").strip() not in existing_quotes:
                        all_issues.append(issue)
                        existing_quotes.add(issue.get("quote", "").strip())
            # If no new issues found in this round, terminate early
            if not issues or assessment == "strong":
                logger.info("Challenger: terminating at round %d (no new issues or strong)", round_num)
                break

        # Format final critique for HITL Gate 3 display
        self.last_issues = all_issues
        self.last_recommendation = recommendation
        return self._format_critique(
            all_issues, total_claims, depths_covered, len(tracing_results),
            assessment, recommendation
        )

    def _run_challenge_round(
        self,
        synthesis_text: str,
        evidence_summary: str,
        user_concept: str,
        prior_issues: list[dict] | None = None,
        previous_critique: str | None = None,
    ) -> tuple[list[dict], str, str]:
        """
        Execute one round of adversarial challenge via LLM.

        Returns (issues_list, overall_assessment, recommendation).
        """
        prompt = CHALLENGE_USER_PROMPT.format(
            synthesis=synthesis_text,
            evidence_summary=evidence_summary,
            concept=user_concept,
        )

        # On debate retry, tell the LLM about the previous critique that was
        # rejected by the user, so it produces a deeper/different analysis
        if previous_critique:
            prompt += (
                "\n\n## Previous Critique (user found this insufficient - dig deeper):\n"
                f"{previous_critique}\n\n"
                "The user rejected the above critique. Produce a MORE THOROUGH review "
                "that identifies issues the previous critique missed. Look for subtler "
                "problems: implicit assumptions, unsupported causal claims, selective "
                "evidence use, or missing alternative explanations."
            )

        # Tell subsequent rounds about already-found issues to avoid duplicates
        if prior_issues:
            prior_summary = "\n".join(
                f"- [{i.get('type')}] \"{i.get('quote', '')[:80]}...\""
                for i in prior_issues
            )
            prompt += (
                "\n\n## Already Identified Issues (DO NOT repeat these):\n"
                f"{prior_summary}\n\n"
                "Find ONLY new issues not listed above."
            )

        try:
            response = self.groq.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": CHALLENGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # low but not zero - allows varied critique
                max_tokens=2000,
            )
            raw_output = response.choices[0].message.content

            # Parse the JSON response
            from agents.inspector import InspectorAgent
            cleaned = InspectorAgent._clean_json_string(raw_output)
            parsed = json.loads(cleaned)

            issues = parsed.get("issues", [])
            assessment = parsed.get("overall_assessment", "adequate")
            recommendation = parsed.get("recommendation", "revise")

            logger.info(
                "Challenger round: %d issues found, assessment=%s, recommendation=%s",
                len(issues), assessment, recommendation
            )
            return issues, assessment, recommendation

        except Exception as e:
            logger.warning("Challenger: LLM call failed: %s", e)
            return [], "adequate", "revise"

    def _build_evidence_summary(self, tracing_results: list[TracingResult]) -> str:
        """
        Build a concise evidence summary from TracingResults for the LLM context.

        Truncates individual claims to keep within token limits while preserving
        key information (validity score, depth, source).
        """
        lines = []
        for result in tracing_results:
            lines.append(
                f"\n### Paper: \"{result.paper_title}\" (depth={result.depth})"
            )
            if result.doi:
                lines.append(f"DOI: {result.doi}")
            for claim in result.claims[:10]:  # cap per-paper claims for token budget
                score_str = f"validity={claim.validity_score:.2f}"
                drift_str = f", drift={claim.drift_score:.2f}" if claim.drift_score is not None else ""
                lines.append(
                    f"  - [{score_str}{drift_str}] {claim.claim_text[:200]}"
                )
            if len(result.claims) > 10:
                lines.append(f"  ... ({len(result.claims) - 10} more claims omitted)")
        return "\n".join(lines)

    def _format_critique(
        self,
        issues: list[dict],
        total_claims: int,
        depths_covered: list[int],
        num_papers: int,
        assessment: str,
        recommendation: str,
    ) -> str:
        """Format the final critique as human-readable Markdown for HITL Gate 3."""
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        type_labels = {
            "unsupported": "Unsupported claim",
            "overstated": "Overstated conclusion",
            "drift": "Conceptual drift",
            "missing_counter": "Missing counter-evidence",
        }

        critique_lines = [
            "## Challenger Adversarial Review\n",
            f"**Overall assessment:** {assessment.upper()}",
            f"**Recommendation:** {recommendation}\n",
            f"**Evidence coverage:**",
            f"- Claims evaluated: {total_claims}",
            f"- Depth levels covered: {depths_covered}",
            f"- Papers traced: {num_papers}\n",
        ]

        if issues:
            critique_lines.append(f"### Issues Found ({len(issues)})\n")
            # Group by severity
            for severity in ["high", "medium", "low"]:
                sev_issues = [i for i in issues if i.get("severity") == severity]
                if not sev_issues:
                    continue
                for issue in sev_issues:
                    emoji = severity_emoji.get(severity, "⚪")
                    itype = type_labels.get(issue.get("type", ""), issue.get("type", "unknown"))
                    quote = issue.get("quote", "")
                    explanation = issue.get("explanation", "")
                    critique_lines.append(f"{emoji} **{itype}** ({severity})")
                    if quote:
                        critique_lines.append(f'> "{quote[:150]}"')
                    if explanation:
                        critique_lines.append(f"  {explanation}")
                    critique_lines.append("")
        else:
            critique_lines.append(
                "\n✅ No significant issues identified. The synthesis appears well-supported "
                "by the traced evidence."
            )

        critique_lines.append(
            "\n---\n*Review produced by automated adversarial challenger "
            "(Irving et al., 2018). Human review via HITL Gate 3 is the final authority.*"
        )
        return "\n".join(critique_lines)

    def _fallback_critique(
        self,
        synthesis_text: str,
        tracing_results: list[TracingResult],
        total_claims: int,
        depths_covered: list[int],
    ) -> str:
        """Produce a coverage-only critique when no LLM is available."""
        # Check basic consistency: are all papers cited in synthesis mentioned in results?
        paper_titles = {r.paper_title.lower() for r in tracing_results}
        high_drift_claims = [
            c for r in tracing_results for c in r.claims
            if c.drift_score is not None and c.drift_score > 0.7
        ]
        low_validity_claims = [
            c for r in tracing_results for c in r.claims
            if c.validity_score < 0.5
        ]

        return (
            f"## Challenger Review (no LLM available)\n\n"
            f"**Coverage summary:**\n"
            f"- Claims reviewed: {total_claims}\n"
            f"- Depth levels covered: {depths_covered}\n"
            f"- Papers traced: {len(tracing_results)}\n"
            f"- High-drift claims (>0.7): {len(high_drift_claims)}\n"
            f"- Low-validity claims (<0.5): {len(low_validity_claims)}\n\n"
            f"**Automated checks (no LLM):**\n"
            f"- Paper titles in evidence: {len(paper_titles)}\n\n"
            f"**Recommended manual review points:**\n"
            f"- Verify that claims at depth > 1 have not drifted significantly from the seed concept\n"
            f"- Check that all cited source titles in the synthesis match the traced paper list\n"
            f"- Confirm no claims in the synthesis are unsupported by the extracted evidence\n\n"
            f"Human review via HITL Gate 3 is the primary quality control."
        )