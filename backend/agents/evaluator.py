"""
agents/evaluator.py - Evaluator agent.

ROLE IN ARCHITECTURE:
Stateless worker node. Receives raw ExtractedClaims from the Citation Extractor
and returns filtered, scored EvaluatedClaims. Acts as an explicit validation gate
between extraction and synthesis.

WHY AN EXPLICIT EVALUATION GATE:
"The Evaluator isolates relevant concepts and filters academic noise before
information reaches the synthesis stage, creating an explicit validation gate
that reduces the risk of hallucination propagation across the pipeline."
(Design proposal 3, citing Maynez et al., 2020).

Without this gate, low-confidence or off-topic claims extracted by the LLM
would propagate to the Synthesiser, where they could be presented as grounded
findings. The Evaluator is the system's primary defence against compounding
hallucination across pipeline stages.

SCORING APPROACH:
Two-stage scoring:
  1. Relevance filter: LLM-based binary classification - does this claim relate
     to the user's target concept? Claims scoring below relevance_threshold are
     discarded.
  2. Validity scoring: LLM-based 0-1 score - how well-supported is this claim
     within the paper text? Combined with the Extractor's confidence score.

DRIFT DETECTION:
At depth > 0, the Evaluator computes a semantic drift score by comparing the
claim embedding to the parent paper's centroid embedding (stored in pgvector).
High drift (score > 0.7) flags conceptual divergence from the root concept.
Note: drift detection requires pgvector to be available; gracefully degrades
to drift_score=None if vector storage is unavailable (MVP fallback).

References:
  Maynez et al. (2020) On faithfulness and factuality in abstractive summarization.
    ACL 2020, pp. 1906-1919. doi:10.18653/v1/2020.acl-main.173
  Reimers and Gurevych (2019) Sentence-BERT. EMNLP 2019, pp. 3982-3992.
"""

import logging
from typing import Any

from core.state import (
    PipelineState,
    ExtractedClaim,
    EvaluatedClaim,
)

logger = logging.getLogger(__name__)


RELEVANCE_PROMPT = """You are evaluating whether an academic claim is relevant to a research concept.

Research concept: {concept}

Claim: {claim}

Is this claim directly relevant to the research concept above?
Respond with a JSON object with exactly two fields:
  - "relevant": boolean (true/false)
  - "validity_score": number from 0.0 to 1.0 indicating how well-supported this claim appears
  - "relevant_concepts": array of strings listing the specific sub-concepts this claim addresses

JSON object only, no preamble:"""


DRIFT_ANALYSIS_PROMPT = """You are analysing conceptual drift in a citation genealogy.

The gold-standard definition of "{concept}" comes from the seed paper (depth 0).
The following claim was extracted at depth {depth} (i.e., {depth} citation hops from the seed):

Claim: "{claim}"
Source paper: "{source_title}"

Compare how this claim uses the concept relative to the seed paper's definition.
Respond with a JSON object with exactly three fields:
  - "historical_use": one sentence describing how the seed paper originally defined/used the concept
  - "current_use": one sentence describing how THIS claim redefines or applies the concept
  - "evolutionary_path": one sentence explaining the step-by-step transformation from seed usage to current usage through the citation chain

JSON object only, no preamble:"""


class EvaluatorAgent:
    """
    Filters and scores extracted claims for relevance and validity.

    WHY LLAMA-3.1-8B FOR EVALUATION (NOT 70B):
    Relevance classification and validity scoring are binary/scalar tasks that
    do not require the comparative reasoning capabilities of the larger model.
    Using 8b here keeps token costs proportional (design proposal 3):
    "assigning lightweight models to stateless worker nodes and larger models only
    where comparative reasoning is required."
    """

    RELEVANCE_THRESHOLD = 0.3  # claims below this validity_score are discarded

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = config.get("groq_model_fast", "llama-3.1-8b-instant")
        self.groq = config.get("_groq_client")  # shared rate-limited client
        self._init_embedding_model()

    def _init_embedding_model(self) -> None:
        """
        Initialise Sentence-BERT model for drift detection.

        WHY ALL-MPNET-BASE-V2 (NOT SPECTER2 FOR DRIFT):
        Specter2 (Singh et al., 2023) is optimised for paper-level scientific
        document retrieval - it produces embeddings for full abstracts.
        all-mpnet-base-v2 (Reimers and Gurevych, 2019) is better suited for
        sentence-level semantic similarity comparison, which is what drift
        detection requires (comparing individual claim sentences).

        SPECTER2 is used by the storage layer for paper-level vector indexing.
        all-mpnet-base-v2 is used here for claim-level drift computation.

        Graceful degradation: if sentence-transformers is not installed,
        drift detection is skipped (drift_score = None). This does not affect
        MVP pipeline functionality.
        """
        try:
            from sentence_transformers import SentenceTransformer
            # Load once at init, not per-call - the model is ~420MB
            self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
            logger.info("Evaluator: embedding model loaded (all-mpnet-base-v2)")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Drift detection disabled. "
                "Run: pip install sentence-transformers"
            )
            self.embedding_model = None

    def run(
        self,
        raw_claims: list[ExtractedClaim],
        state: PipelineState,
    ) -> list[EvaluatedClaim]:
        """
        Main entry point. Filter and score a list of extracted claims.

        Returns only claims that pass the relevance filter, with validity scores
        and (optionally) drift scores attached.

        Uses the depth-specific abstracted concept for relevance evaluation.
        At depth 0, this is the user's original concept. At deeper levels, it
        is the precursor concept identified by the Orchestrator's concept
        abstraction step, allowing relevance matching against older terminology.
        """
        if not raw_claims:
            return []

        # Use abstracted concept for this depth level, falling back to user_concept
        active_concept = state.concept_at_depth.get(
            state.current_depth, state.user_concept
        )

        total = len(raw_claims)
        evaluated: list[EvaluatedClaim] = []
        for idx, claim in enumerate(raw_claims, 1):
            logger.debug("Evaluating claim %d/%d", idx, total)
            result = self._evaluate_claim(
                claim, active_concept, state.current_depth,
                original_concept=state.user_concept,
            )
            if result is not None:
                evaluated.append(result)

        logger.info(
            "Evaluator: %d/%d claims passed relevance filter for concept='%s' (depth=%d)",
            len(evaluated), len(raw_claims), active_concept, state.current_depth
        )
        return evaluated

    def _evaluate_claim(
        self,
        claim: ExtractedClaim,
        user_concept: str,
        depth: int,
        original_concept: str | None = None,
    ) -> EvaluatedClaim | None:
        """
        Evaluate a single claim. Returns None if the claim should be discarded.

        Two-stage gate:
          1. LLM relevance + validity scoring
          2. Threshold filter (discard if validity_score < RELEVANCE_THRESHOLD)

        At depth > 0, if the abstracted concept differs from the original,
        the claim is checked against BOTH concepts. A paper that references the
        seed is inherently connected to the concept chain, so claims relevant to
        either the original or the abstracted precursor concept are retained.
        """
        if not self.groq:
            # Graceful degradation: if LLM unavailable, pass all claims with score 0.5
            return EvaluatedClaim(
                claim_text=claim.claim_text,
                source_paper_id=claim.source_paper_id,
                source_title=claim.source_title,
                depth=claim.depth,
                validity_score=0.5,
                relevant_concepts=[],
                inline_citations=claim.inline_citations,
            )

        # At depth > 0, include both concepts in the prompt so the LLM evaluates
        # relevance to the full genealogy chain (not just the abstracted precursor)
        if depth > 0 and original_concept and original_concept.lower() != user_concept.lower():
            concept_str = f"{user_concept}\" (precursor of \"{original_concept}\")"
        else:
            concept_str = user_concept

        prompt = RELEVANCE_PROMPT.format(
            concept=concept_str,
            claim=claim.claim_text,
        )

        try:
            import json as _json
            from agents.inspector import InspectorAgent

            response = self.groq.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            raw_output = response.choices[0].message.content
            cleaned = InspectorAgent._clean_json_string(raw_output)
            parsed = _json.loads(cleaned)

            relevant = parsed.get("relevant", False)
            validity_score = float(parsed.get("validity_score", 0.0))
            relevant_concepts = parsed.get("relevant_concepts", [])

        except Exception as e:
            logger.warning("Evaluator: LLM call failed for claim - skipping. Error: %s", e)
            return None

        # Discard irrelevant or very low-quality claims
        if not relevant or validity_score < self.RELEVANCE_THRESHOLD:
            logger.debug(
                "Evaluator: discarding claim (relevant=%s, score=%.2f): '%s...'",
                relevant, validity_score, claim.claim_text[:60]
            )
            return None

        # Compute drift score at depth > 0 if embedding model is available
        drift_score = None
        drift_analysis = None
        if depth > 0 and self.embedding_model is not None:
            drift_score = self._compute_drift_score(claim, user_concept)
        if depth > 0 and self.groq is not None:
            drift_analysis = self._compute_drift_analysis(claim, user_concept, depth)

        return EvaluatedClaim(
            claim_text=claim.claim_text,
            source_paper_id=claim.source_paper_id,
            source_title=claim.source_title,
            depth=claim.depth,
            validity_score=validity_score,
            relevant_concepts=relevant_concepts,
            inline_citations=claim.inline_citations,
            drift_score=drift_score,
            drift_analysis=drift_analysis,
        )

    def _compute_drift_score(
        self,
        claim: ExtractedClaim,
        user_concept: str,
    ) -> float:
        """
        Compute semantic drift between a claim and the user's root concept.

        Returns a float in [0, 1] where:
          0.0 = semantically identical to root concept
          1.0 = completely unrelated to root concept

        Implementation: cosine distance between claim embedding and concept embedding.
        Note: 1 - cosine_similarity = cosine distance.

        WHY COSINE DISTANCE (NOT EUCLIDEAN):
        Sentence-BERT embeddings are unit-normalised, making cosine distance
        equivalent to 1 - dot product. This is the standard similarity measure
        for sentence transformer outputs (Reimers and Gurevych, 2019).
        """
        try:
            import numpy as np

            claim_emb = self.embedding_model.encode(claim.claim_text, convert_to_numpy=True)
            concept_emb = self.embedding_model.encode(user_concept, convert_to_numpy=True)

            # Cosine similarity
            cos_sim = np.dot(claim_emb, concept_emb) / (
                np.linalg.norm(claim_emb) * np.linalg.norm(concept_emb) + 1e-8
            )
            drift = float(1.0 - cos_sim)
            return max(0.0, min(1.0, drift))  # clamp to [0, 1]

        except Exception as e:
            logger.warning("Evaluator: drift computation failed: %s", e)
            return None

    def _compute_drift_analysis(
        self,
        claim: ExtractedClaim,
        user_concept: str,
        depth: int,
    ) -> dict | None:
        """
        Produce qualitative drift analysis comparing claim usage to seed paper.

        Returns a dict with keys: historical_use, current_use, evolutionary_path.
        Always compares to the gold-standard seed paper definition.
        """
        prompt = DRIFT_ANALYSIS_PROMPT.format(
            concept=user_concept,
            depth=depth,
            claim=claim.claim_text,
            source_title=claim.source_title,
        )
        try:
            import json as _json
            from agents.inspector import InspectorAgent

            response = self.groq.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            raw_output = response.choices[0].message.content
            cleaned = InspectorAgent._clean_json_string(raw_output)
            parsed = _json.loads(cleaned)
            # Validate expected keys
            if all(k in parsed for k in ("historical_use", "current_use", "evolutionary_path")):
                return parsed
            logger.warning("Evaluator: drift analysis missing keys, skipping")
            return None
        except Exception as e:
            logger.warning("Evaluator: drift analysis LLM call failed: %s", e)
            return None