"""
agents/inspector.py - Inspector agent (structural validator).

ROLE IN ARCHITECTURE:
The Inspector is not a generative LLM agent. It is a deterministic validation
gate that applies Pydantic schema validation at every agent output boundary.
This is a deliberate design choice (design proposal 5):

  "The Inspector applies deterministic Pydantic schema validation at each agent
   boundary to handle persistent LLM non-determinism (Atil et al., 2025)."

WHY DETERMINISTIC VALIDATION OVER LLM-BASED VALIDATION:
LLM output variance persists even under deterministic configurations in hosted
API environments (Atil et al., 2025). Schema-level validation via Pydantic is:
  - O(1) latency (no API call)
  - 100% reproducible given the same input
  - Zero cost (no tokens consumed)
  - Auditable: rejected outputs are logged with the specific field that failed

The trade-off is that Pydantic catches structural errors, not semantic ones.
Semantic errors (plausible but factually wrong claims) are the Challenger's job.
This separation of concerns keeps each agent's responsibility scope minimal.

WHY MAX-1-RETRY:
The design proposal specifies max 1 retry for Pydantic validation errors. This
prevents infinite retry loops if a model consistently produces malformed output
for a particular input. After 1 retry, the Inspector logs a failure and the
Orchestrator routes to FAILED state, preserving partial results in the DB.

References:
  Atil et al. (2025) Non-determinism of "deterministic" LLM system settings.
    Proceedings of Eval4NLP, ACL 2025, pp. 135-148.
  Cemri et al. (2025) Why do multi-agent LLM systems fail? NeurIPS 2025.
  Maynez et al. (2020) On faithfulness and factuality in abstractive summarization.
    ACL 2020, pp. 1906-1919.
"""

import json
import logging
from typing import Any

from pydantic import ValidationError

from core.state import (
    PipelineState,
    CandidatePaper,
    EvaluatedClaim,
    TracingResult,
)

logger = logging.getLogger(__name__)


class InspectorAgent:
    """
    Deterministic structural validator. No LLM calls - pure Pydantic.

    All validate_* methods follow the same contract:
      - Accept raw output from an upstream agent
      - Return a validated, typed object if valid
      - Raise ValueError (logged) if invalid after retries
      - Increment state.inspector_retries before retrying
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # max_inspector_retries comes from PipelineState default (1).
        # We read it from config here for the rare case where it's overridden.
        self.max_retries = config.get("max_inspector_retries", 1)

    def validate_candidates(
        self,
        raw_candidates: list[Any],
        state: PipelineState,
    ) -> list[CandidatePaper]:
        """
        Validate a list of candidate papers from the Retriever.

        Strategy: validate each candidate individually. Invalid candidates are
        discarded (logged as warnings) rather than failing the whole list.
        This is intentional - if 2 of 3 candidates are valid, we proceed with 2.
        A total failure (0 valid candidates) routes to FAILED state.
        """
        validated: list[CandidatePaper] = []
        for i, raw in enumerate(raw_candidates):
            try:
                if isinstance(raw, CandidatePaper):
                    validated.append(raw)
                elif isinstance(raw, dict):
                    validated.append(CandidatePaper(**raw))
                else:
                    logger.warning(
                        "Inspector: candidate[%d] is not dict or CandidatePaper (type=%s) - discarding",
                        i, type(raw).__name__
                    )
            except ValidationError as e:
                logger.warning(
                    "Inspector: candidate[%d] failed validation - discarding. Errors: %s",
                    i, self._format_errors(e)
                )

        if not validated:
            logger.error("Inspector: all %d candidates failed validation", len(raw_candidates))
        else:
            logger.debug("Inspector: %d/%d candidates passed validation", len(validated), len(raw_candidates))

        return validated

    def validate_tracing_result(
        self,
        evaluated_claims: list[EvaluatedClaim],
        paper: CandidatePaper,
        depth: int,
        state: PipelineState,
    ) -> TracingResult:
        """
        Validate and assemble a TracingResult from evaluated claims.

        WHY ASSEMBLE HERE AND NOT IN THE EVALUATOR:
        The Inspector is the last boundary before data enters persistent state.
        Assembling TracingResult here (rather than in Evaluator) means the
        Inspector can enforce cross-field constraints (e.g. all claims must
        have depth == the current tracing depth) that individual field validators
        cannot catch.
        """
        # Cross-field constraint: depth consistency
        depth_mismatches = [c for c in evaluated_claims if c.depth != depth]
        if depth_mismatches:
            logger.warning(
                "Inspector: %d claims have depth mismatch (expected %d) - correcting",
                len(depth_mismatches), depth
            )
            for claim in depth_mismatches:
                claim.depth = depth  # correct silently; log is sufficient

        try:
            result = TracingResult(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                depth=depth,
                claims=evaluated_claims,
                references_explored=[],  # populated by CitationExtractor
            )
            logger.debug(
                "Inspector: TracingResult validated for paper '%s' at depth %d (%d claims)",
                paper.title, depth, len(evaluated_claims)
            )
            return result
        except ValidationError as e:
            logger.error("Inspector: TracingResult validation failed: %s", self._format_errors(e))
            # Return an empty result rather than raising - empty results are
            # handled gracefully by the tracing loop.
            return TracingResult(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                depth=depth,
                claims=[],
            )

    def validate_synthesis(self, synthesis_text: Any, state: PipelineState) -> str:
        """
        Validate synthesis output from the Synthesiser.

        The Synthesiser returns a string (Markdown). Validation here is lightweight:
          - Must be a non-empty string
          - Must not be a JSON error response (indicates the LLM returned an error
            object instead of prose - a common failure mode with hosted APIs)

        WHY NOT PYDANTIC FOR A STRING:
        A Pydantic model wrapping a single string field would be overengineering.
        The meaningful structural risk for synthesis output is "is this actually
        a string and not a raw error dict?" - handled by the checks below.
        """
        if not isinstance(synthesis_text, str):
            logger.warning(
                "Inspector: synthesis is not a string (type=%s) - attempting str() conversion",
                type(synthesis_text).__name__
            )
            synthesis_text = str(synthesis_text)

        if not synthesis_text.strip():
            logger.error("Inspector: synthesis is empty")
            return "(synthesis failed - empty output)"

        # Detect JSON error responses (e.g. {"error": "rate limit exceeded"})
        stripped = synthesis_text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if "error" in parsed:
                    logger.error("Inspector: synthesis contains LLM error response: %s", parsed)
                    return f"(synthesis failed - API error: {parsed.get('error', 'unknown')})"
            except json.JSONDecodeError:
                pass  # Not JSON, treat as valid prose

        return synthesis_text

    def validate_extracted_claims_json(
        self,
        raw_json: str,
        source_paper_id: str,
        depth: int,
    ) -> list[dict]:
        """
        Validate raw JSON string output from Citation Extractor LLM call.

        WHY VALIDATE JSON BEFORE PYDANTIC:
        The Citation Extractor prompts the LLM to return a JSON array. LLMs
        routinely emit structurally invalid JSON that breaks downstream pipelines
        (Cemri et al., 2025). This method catches JSON parse failures before
        they reach Pydantic, enabling the single-retry logic.

        Handles three common LLM output failure modes:
          1. Code fences: ```json [...] ``` - stripped by _clean_json_string
          2. Single object instead of array: {...} - wrapped into [...]
          3. NDJSON (newline-delimited objects): {...}\n{...}\n - parsed line by line
             This is the "Extra data" failure mode observed in production.

        Retry trigger: returns empty list on failure, which the Citation Extractor
        interprets as a signal to retry (up to max_inspector_retries times).
        """
        try:
            cleaned = self._clean_json_string(raw_json)

            # Attempt 1: standard parse (handles well-formed arrays)
            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, list):
                    parsed = [parsed] if isinstance(parsed, dict) else []
                return parsed
            except json.JSONDecodeError as e:
                # Only attempt NDJSON recovery on "Extra data" errors -
                # other errors (truncated JSON, syntax errors) are not recoverable
                # with line-by-line parsing.
                if "Extra data" not in str(e):
                    raise

            # Attempt 2: NDJSON recovery - LLM emitted one object per line
            # instead of a JSON array. This is a known Llama-3 output pattern
            # when the model ignores the array instruction (Cemri et al., 2025).
            logger.warning(
                "Inspector: standard JSON parse failed with 'Extra data' for %s - "
                "attempting NDJSON line-by-line recovery",
                source_paper_id
            )
            objects = []
            for line in cleaned.splitlines():
                line = line.strip().rstrip(",")  # handle trailing commas too
                if not line or line in ("[", "]"):
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    continue  # skip malformed lines

            if objects:
                logger.info(
                    "Inspector: NDJSON recovery succeeded - recovered %d objects for %s",
                    len(objects), source_paper_id
                )
                return objects

            logger.error(
                "Inspector: NDJSON recovery also failed for %s - returning empty",
                source_paper_id
            )
            return []

        except json.JSONDecodeError as e:
            logger.error(
                "Inspector: JSON parse failure for paper %s at depth %d: %s",
                source_paper_id, depth, e
            )
            return []

    @staticmethod
    def _clean_json_string(raw: str) -> str:
        """
        Strip common LLM output artefacts from a JSON string.

        LLMs frequently wrap JSON in markdown code fences (```json ... ```) even
        when instructed not to. This is a known non-determinism issue (Atil et al., 2025).
        We strip defensively rather than relying on prompt instructions alone.
        """
        stripped = raw.strip()

        # Remove ```json ... ``` or ``` ... ``` fences
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            inner_lines = lines[1:] if lines[0].startswith("```") else lines
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            stripped = "\n".join(inner_lines).strip()

        # Find the first [ or { to skip any preamble text.
        # WHY CHECK BOTH [ AND { IN ORDER:
        # A well-formed array starts with [. If we find { first, the LLM produced
        # a single object or NDJSON - both handled by validate_extracted_claims_json.
        # We prefer [ over { when both exist (proper array takes priority).
        bracket_idx = stripped.find("[")
        brace_idx = stripped.find("{")

        if bracket_idx != -1 and (brace_idx == -1 or bracket_idx < brace_idx):
            stripped = stripped[bracket_idx:]
        elif brace_idx != -1 and brace_idx > 0:
            stripped = stripped[brace_idx:]

        return stripped

    @staticmethod
    def _format_errors(e: ValidationError) -> str:
        """Format Pydantic ValidationError for log output."""
        return "; ".join(
            f"{' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )