# Reflection Issues Log - Concept Genealogy Pipeline

## Reflective Framework: Borton/Rolfe Model (What / So What / Now What)

### Research Themes for Deployment Reflection
1. **Responsible Deployment - Emergent Behaviours**
2. **AI Safety Constraints**
3. **Explainable AI (XAI)**

---

## Issue 1: Pipeline Thread Produced No Logs in Frontend

| Dimension | Detail |
|-----------|--------|
| **What** | After launching a pipeline run via the web UI, the frontend polling showed no log entries despite the pipeline executing successfully in the backend thread. The `RunLogHandler` was not capturing output from the pipeline's background thread. |
| **Root Cause** | The logging handler was attached to the wrong logger scope, so messages emitted by agents in the child thread were not routed to the `RunStore` log buffer. |
| **Fix** | Attached the `RunLogHandler` to the root logger for the run's thread, ensuring all agent-level log messages (retriever, inspector, synthesiser, challenger) were captured and exposed via the `/runs/{id}/logs` polling endpoint. |

**So What - Emergent Behaviours:**
This is a classic emergent behaviour in multi-threaded AI systems. The pipeline's orchestration (LangGraph StateGraph) spawns execution across multiple nodes, each emitting logs independently. The *emergent* property is that observability vanishes silently - the system appears to work (returns results) but the human operator loses all visibility into intermediate reasoning. In a responsible deployment, silent failures in observability are dangerous because they prevent the human from detecting when the AI has taken an unexpected path.

**Now What:**
Ensure logging infrastructure is tested at the integration level (not just unit level). Add a smoke test that verifies log entries are captured from within a threaded pipeline execution.

---

## Issue 2: Gate 3 Rejection Routed to Synthesiser Instead of Challenger

| Dimension | Detail |
|-----------|--------|
| **What** | When a user rejected the final output at HITL Gate 3, the pipeline routed back to the Synthesiser node (re-synthesis) instead of back to the Challenger node (re-challenge). This contradicted the system's activity diagram which specifies Gate 3 rejection should trigger deeper adversarial review. |
| **Root Cause** | The conditional edge `_route_after_final_review` returned `"synthesise"` on rejection. This was a design implementation error - the original code assumed rejection meant the synthesis was wrong, not that the critique was insufficient. |
| **Fix** | Changed `_route_after_final_review` to return `"challenge"` on rejection. Added `challenger_retries` counter with `max_challenger_retries` cap. The Challenger now receives `previous_critique` on retry so it can dig deeper rather than repeating the same analysis. |

**So What - AI Safety Constraints:**
This issue illustrates a violation of the *bounded autonomy* safety constraint. The pipeline's HITL gates exist to ensure humans can steer AI behaviour at critical decision points (Mosqueira-Rey et al., 2023). When Gate 3 rejection routes to the wrong node, the user's explicit safety intervention (rejection) produces an unintended system response. The AI effectively ignores the user's corrective signal. This is a *misaligned feedback loop* - the human signals "go deeper on critique" but the system hears "redo the whole synthesis."

**Now What:**
Graph routing logic (conditional edges) should be validated against the formal activity diagram as a design artefact. Consider adding assertion-based tests that verify each HITL gate rejection routes to the architecturally-specified recovery node.

---

## Issue 3: SJR Ranking Not Displayed at Gate 1 (Seed Selection)

| Dimension | Detail |
|-----------|--------|
| **What** | Gate 1 presented candidate papers without journal quality indicators. Users had no signal for whether a paper came from a high-impact venue, making seed selection opaque. The SCImago Journal Rank (SJR) data was available in a CSV but not integrated. |
| **Root Cause** | `_parse_paper()` was a static method with no access to the SJR index. The venue name from the Semantic Scholar API (`publicationVenue.name`) was never cross-referenced against the loaded SJR data. |
| **Fix** | Converted `_parse_paper()` to an instance method. Added SJR lookup by venue name during paper parsing. Handled European decimal notation (`,` → `.`) in the CSV. Added `sjr_score: Optional[float]` to `CandidatePaper`. Frontend displays a purple SJR badge conditionally. |

**So What - Explainable AI:**
This directly addresses XAI principles. The pipeline's Retriever agent makes a ranking decision (which papers to present), but without SJR scores the user cannot understand *why* a paper was surfaced or judge its authority. Displaying SJR scores makes the retrieval decision partially explainable - the user can now reason about whether the system's candidate selection aligns with their quality expectations. This supports *interpretability at decision boundaries* (Guidotti et al., 2018).

**Now What:**
Consider adding additional explainability signals at Gate 1: citation velocity, recency weighting rationale, or a brief "why this paper?" annotation generated by the Retriever. Each additional signal reduces opacity in the AI's candidate selection.

---

## Issue 4: Rate-Limit Retries Invisible to User

| Dimension | Detail |
|-----------|--------|
| **What** | When the Elsevier/Semantic Scholar API returned HTTP 429 (rate limited), the retriever silently retried with exponential backoff. The frontend showed no indication of progress - the UI appeared frozen for 30-60 seconds with no feedback. |
| **Root Cause** | The `_get()` method in the retriever handled retries internally without emitting any observable event to the progress callback or log stream. |
| **Fix** | Added progress callback emissions on 429 retries: `"Rate limited - waiting {wait}s before retry {attempt}/{max}..."`. Frontend parses these messages into a styled amber-pulse waiting indicator with the specific wait duration. |

**So What - Emergent Behaviours:**
Rate-limiting is an *environmental emergent behaviour* - it arises from the interaction between the AI system and external API infrastructure, not from the AI's own logic. The system's behaviour (appearing frozen) emerges from an external constraint that the original design did not surface to the user. In deployment, users interpret unresponsive AI as broken, potentially abandoning valid runs. This is a form of *silent degradation* that responsible deployment must address through transparency.

**Now What:**
Audit all external API interactions for similar silent-retry patterns. Consider a generalised progress/status callback pattern for any operation that may block on external dependencies (API calls, database connections, file I/O).

---

## Issue 5: Gate 2 Did Not Show Source References

| Dimension | Detail |
|-----------|--------|
| **What** | HITL Gate 2 (Synthesis Approval) displayed only the synthesis text. Users had no way to verify which papers the synthesis was based on, their depth in the citation chain, or how many claims were extracted from each. They had to approve the synthesis "blind" without seeing the evidence base. |
| **Root Cause** | `gate_2_synthesis_approval()` only received and passed `synthesis_text`. The `tracing_results` were available in `PipelineState` but never included in the `GateRequest` payload sent to the frontend. |
| **Fix** | Added `tracing_papers: list[dict]` to `GateRequest`. The orchestrator now extracts paper metadata (title, authors, depth, DOI, claim count) from `state.tracing_results` and passes it to Gate 2. Frontend renders a styled reference list below the synthesis showing `[n] Title - Authors (Depth X, N claims) DOI: link`. |

**So What - Explainable AI:**
This is a critical XAI gap. The synthesis is the AI's primary output, but without showing *which evidence* produced it, the user cannot evaluate whether the AI's reasoning chain is sound. Displaying the traced papers with their depth and claim counts provides *provenance transparency* - the user can now trace the synthesis back to its evidence sources. This aligns with the XAI principle of *source attribution* (Arrieta et al., 2020) and enables the user to identify if the synthesis over-relies on a single paper or ignores deep-chain evidence.

**Now What:**
Consider adding inline citation markers within the synthesis text itself (e.g., `[1]`, `[2]`) that map to the reference list. This would provide *claim-level* rather than *document-level* explainability, showing exactly which paper supports each synthesised statement.

---

## Issue 6: Challenger Required Manual Re-Challenge Instead of Auto-Iterating

| Dimension | Detail |
|-----------|--------|
| **What** | The Challenger's adversarial review ran 2 internal debate rounds, then immediately presented results at Gate 3. If the critique was shallow (e.g., only 2 high-severity issues found), the user had to manually reject at Gate 3 to trigger a re-challenge, which then found additional issues (e.g., 3 high-severity items). The system required unnecessary human intervention for what should be an automatic deepening of analysis. |
| **Root Cause** | The Challenger's `run()` method returned after `MAX_DEBATE_ROUNDS` (2) without evaluating whether its own critique was thorough. The `recommendation` field ("approve"/"revise"/"reject") was computed but never used to drive further iteration. It was a local variable, not exposed to the orchestrator. |
| **Fix** | Added `self.last_recommendation` attribute to `ChallengerAgent`. Modified `_node_challenge` to implement an auto-retry loop: if the Challenger's own recommendation is "revise", it automatically re-runs with `previous_critique` (up to 2 auto-retries, so 3 total passes maximum). Gate 3 only appears once the Challenger is satisfied or the retry cap is hit. |

**So What - AI Safety Constraints & Emergent Behaviours:**
This issue sits at the intersection of safety and emergent behaviour. The Challenger is a *safety mechanism* - it exists to catch flaws in the synthesis before human approval. When it produces a shallow critique and immediately defers to the human, it shifts the safety burden back to the user. The *emergent behaviour* is that the Challenger "knows" its critique is incomplete (recommendation = "revise") but fails to act on that self-assessment.

The fix implements *self-critical iteration* - the AI agent uses its own meta-cognitive signal (the recommendation) to determine whether it has been thorough enough. This is a form of *bounded self-improvement* that respects safety constraints: the auto-retry is capped (max 2 additional passes) to prevent infinite loops, and the human still has final approval at Gate 3.

**Now What:**
Monitor whether the auto-retry loop produces diminishing returns (same issues repeated across iterations). Consider adding a convergence check: if two consecutive runs produce identical issue sets, terminate early regardless of the recommendation field. This prevents resource waste on non-productive self-critique loops.

---

## Issue 7: UX Improvements for Challenger Display at Gate 3

| Dimension | Detail |
|-----------|--------|
| **What** | The Challenger's critique at Gate 3 was displayed as raw text without structure. Users had difficulty parsing severity levels, specific issue locations, and the overall assessment. |
| **Fix** | Structured the Gate 3 display with severity-grouped issues (High/Medium/Low), clear visual hierarchy, and explicit reject button labelling ("✗ Reject (Re-challenge)") to make the re-challenge action discoverable. |

**So What - Explainable AI:**
Presenting adversarial review results without structure forces the user to perform their own analysis of the AI's analysis - a cognitive burden that undermines the HITL gate's purpose. Structured presentation with severity grouping provides *decision-relevant transparency*: the user immediately sees the most critical issues first and can make an informed approve/reject decision. The button label "Re-challenge" also provides *action transparency* - the user knows what their rejection will trigger.

---

## Issue 8: Depth-Adaptive Concept Abstraction Failure

| Dimension | Detail |
|-----------|--------|
| **What** | The Evaluator's relevance filter used the original user concept (e.g., "attention mechanism") at all tracing depths. At depth 2+, ancestral papers used older terminology (e.g., "selective attention", "feature-based saliency") and were systematically discarded as irrelevant - the "score < threshold → Discard paper" path in the activity diagram fired incorrectly for valid ancestral papers. |
| **Root Cause** | The concept matching was static: a single string compared against extracted claims at every depth. Since academic terminology evolves over decades, a 1990s paper on "selective attention" has no lexical overlap with "attention mechanism" despite being its direct intellectual ancestor. |
| **Fix** | Implemented depth-adaptive concept abstraction: at each depth transition, the LLM identifies the precursor concept from extracted claims at the current depth. The Evaluator uses this abstracted term for relevance matching at the next depth. The concept evolution chain is stored in state (`concept_at_depth`) and visualised as edge labels on the genealogy graph. |

**So What - Emergent Behaviours:**
This is an emergent failure arising from the interaction between static matching logic and the temporal properties of academic language evolution. No single component is "broken" - the Evaluator correctly rejects papers that don't match the concept, and older papers correctly use different terminology. The failure *emerges* from their composition across time. This is analogous to what Cemri et al. (2025) describe as compositional failures in multi-agent systems: each agent behaves correctly in isolation, but their sequential interaction produces incorrect system-level behaviour. The system appeared to work (it traced papers at depth 1 successfully) but silently truncated the genealogy at deeper levels.

**Now What:**
Any pipeline that operates across temporal domains (historical data, evolving terminology, versioned APIs) must account for concept drift at the system design level, not as an afterthought. The abstraction step should be a first-class pipeline node validated by the Inspector, not an implicit transformation buried in the Evaluator.

---

## Issue 9: Semantic Scholar Reference Index Gaps Required CrossRef Fallback

| Dimension | Detail |
|-----------|--------|
| **What** | The activity diagram's "recurse: next DOI" path assumed that reference resolution via Semantic Scholar would always succeed. In practice, S2's `/references` endpoint returned empty arrays for papers with known extensive bibliographies (e.g., highly-cited papers from publishers like Elsevier and Wiley that deposit reference metadata selectively). The recursive tracing loop terminated prematurely at depth 1 for affected papers. |
| **Root Cause** | Semantic Scholar's reference index has coverage gaps - publisher-deposited metadata does not always include structured reference lists. The pipeline had a single point of failure: one API for reference resolution. |
| **Fix** | Implemented CrossRef as the primary reference source. CrossRef receives publisher-deposited reference metadata directly from major publishers (Elsevier, Springer, Wiley), providing near-complete DOI coverage. Reference DOIs are fetched from CrossRef first (no rate limit), then batch-resolved to S2 paper IDs via the `/paper/batch` endpoint. S2 `/references` is used as fallback only when no DOI is available for a paper. |

**So What - Emergent Behaviours:**
This is an *environmental emergent behaviour* - the pipeline's failure mode depends on which specific paper is being traced, which publisher deposited its metadata, and whether that publisher included structured references in their S2 submission. The system cannot predict in advance which papers will trigger the gap. From a responsible deployment perspective, this creates *non-reproducible partial results*: the same pipeline run on different papers may trace to depth 3 or silently stop at depth 1, with no error signal to distinguish "no references exist" from "references exist but the API doesn't know about them." The user cannot distinguish a genuinely terminal citation chain from a data coverage failure.

**Now What:**
External API dependencies in multi-agent pipelines must be treated as unreliable by default. Design patterns should include: (1) multiple data source fallback chains, (2) explicit "data unavailable" signals distinct from "data does not exist", and (3) coverage metrics that report what fraction of expected references were actually resolved.

---

## Issue 10: No Valid Gold Standard Notification - Silent Failure

| Dimension | Detail |
|-----------|--------|
| **What** | The activity diagram explicitly specifies a "No valid Gold Standard found → Notify user" path after the Retriever's search. In the implementation, when no papers matched the user's concept, the Retriever returned an empty candidate list. The HITL Gate 1 presented 0 candidates with no explanatory message - the user saw an empty selection interface with no guidance on why or what to do next. |
| **Root Cause** | The Retriever treated "no results" as a valid state (empty list) rather than as a distinct failure mode requiring a different user communication path. There was no conditional check between retrieval and gate presentation. |
| **Fix** | Added a conditional check after retrieval: if candidates list is empty, the pipeline emits a specific error message explaining that no suitable seed paper was found for the given concept, suggests reformulating the query (broader terms, alternative spellings), and does not present the gate. The user is returned to the input form rather than shown an empty selection. |

**So What - AI Safety Constraints:**
This violates the *informed consent* principle of HITL design (Mosqueira-Rey et al., 2023). A HITL gate's purpose is to give the human meaningful decision authority. Presenting an empty gate with no context forces the user to make a decision (or non-decision) without understanding the system's state. In safety terms, this is a *silent degradation of human oversight* - the gate technically exists, but it cannot fulfil its function. The user may interpret the empty list as "the system is loading" or "the system is broken" rather than "no valid papers exist for this concept." In a responsible deployment, every failure path must be communicated with sufficient context for the human to take corrective action.

**Now What:**
Audit all HITL gates for edge cases where the gate's payload is empty or degenerate. Each gate should have a minimum viable information threshold below which it presents a diagnostic explanation rather than a decision interface. This aligns with the "meaningful human control" criterion - gates must provide enough information for the human's intervention to be genuinely informed.

---

## Issue 11: pgvector Drift Detection Non-Functional Despite Schema Existing

| Dimension | Detail |
|-----------|--------|
| **What** | The database schema included a `vector(768)` column on the `traced_claims` table and the Evaluator had code paths referencing cosine similarity for drift scoring. However, embeddings were never actually computed or stored - the vector column contained only NULL values. The drift detection feature appeared functional in code review but produced no results at runtime. All drift scores defaulted to 0.0, making the Evaluator's "conceptual drift detected" branch unreachable. |
| **Root Cause** | The `sentence-transformers` model (`all-mpnet-base-v2`) was listed in `requirements.txt` and imported in the Evaluator, but the embedding computation was gated behind a condition that checked for an existing embedding (to avoid recomputation) - since no embeddings existed, the "skip if already computed" check always passed, and no embeddings were ever generated. A classic initialisation deadlock: the system waits for state that only it can create. |
| **Fix** | Added explicit embedding computation during claim persistence: when a new claim is stored, its embedding is computed immediately and stored in the vector column. Added a backfill routine for existing claims. Verified drift scores now produce non-zero values for semantically divergent claims across depths. |

**So What - Emergent Behaviours:**
This is a *silent capability degradation* - the system's architecture promises a feature (semantic drift detection) that is structurally present but operationally inert. The emergent property is that no single component fails: the schema is correct, the model loads, the Evaluator's drift logic is syntactically valid. The failure emerges from the interaction between the persistence layer (never writing) and the computation layer (never triggered because nothing to compare against). This is particularly dangerous in responsible deployment because it creates a false sense of security - a code review or architecture diagram suggests drift detection exists, but no actual AI safety benefit is being provided. Stakeholders may rely on a capability that is non-functional.

**Now What:**
Critical AI safety features (drift detection, adversarial review, validation gates) must have integration tests that verify end-to-end functionality, not just component correctness. A test that asserts "drift score > 0 when claims are semantically divergent" would have caught this immediately. Feature capability should be verified through observable behaviour, not structural presence.

---

## Issue 12: Challenger Agent Was a No-Op Stub - Safety Mechanism Absent

| Dimension | Detail |
|-----------|--------|
| **What** | The activity diagram specifies "Challenger: analyse conceptual drift" as a mandatory pipeline node with debate retry logic (up to 3 rounds). The initial implementation contained only a stub: a LangGraph node that logged "Challenger not yet implemented" and passed state through unchanged. The pipeline appeared to complete successfully - the Synthesiser's output was delivered to the user without any adversarial review. |
| **Root Cause** | Incremental development prioritised the happy path (retrieval → extraction → synthesis). The Challenger was deferred to Week 3 as a "stretch" item. However, the LangGraph graph included the node in the execution path, meaning runs completed without the safety mechanism and without any indication that adversarial review was skipped. |
| **Fix** | Implemented the full Challenger agent: structured adversarial critique using llama-3.3-70b, multi-round self-critical iteration (MAX_DEBATE_ROUNDS=2 with auto-retry on "revise" recommendation), JSON-structured output with issue severity/type/quote/explanation, overall assessment, and graceful fallback to coverage-only review when LLM fails. Gate 3 (Final Approval) now presents the Challenger's critique alongside the synthesis. |

**So What - AI Safety Constraints:**
This is a direct violation of the *defence in depth* principle in AI safety. The Challenger exists as an adversarial safety layer - its purpose is to identify hallucinations, unsupported claims, and logical gaps in the synthesis before the user accepts it (Irving, Christiano and Amodei, 2018). When this node is a no-op, the entire pipeline's safety posture collapses to a single layer: the user's own ability to evaluate the synthesis without AI assistance. The system *appeared* safe (the node existed in the graph, runs completed, output was delivered) but the adversarial review - the mechanism that justified reduced scrutiny by the user - was absent.

This is the AI safety equivalent of a smoke detector with no battery: the physical device is present, compliance appears met, but no actual protection is provided. In responsible deployment, a missing safety mechanism must either block the pipeline (fail-safe) or clearly communicate to the user that adversarial review was skipped (fail-transparent).

**Now What:**
Safety-critical pipeline nodes should have a *capability assertion* at startup: if the node cannot perform its function (model unavailable, logic not implemented, dependencies missing), it must either halt the pipeline or inject an explicit warning into the output that reaches the user. "Silent pass-through" should never be the default behaviour for a safety mechanism. Consider implementing a `safety_mechanism_active: bool` flag in the pipeline state that gates whether the output is labelled as "adversarially reviewed" or "unreviewed."

---

## Cross-Cutting Analysis

### Theme Map

| Issue | Emergent Behaviours | AI Safety Constraints | Explainable AI |
|-------|:---:|:---:|:---:|
| 1. No logs in frontend | ✓ | | |
| 2. Wrong Gate 3 routing | | ✓ | |
| 3. Missing SJR scores | | | ✓ |
| 4. Invisible rate-limiting | ✓ | | |
| 5. No references at Gate 2 | | | ✓ |
| 6. Manual re-challenge | ✓ | ✓ | |
| 7. Unstructured Gate 3 display | | | ✓ |
| 8. Concept abstraction failure | ✓ | | |
| 9. S2 reference index gaps | ✓ | | |
| 10. No Gold Standard notification | | ✓ | |
| 11. pgvector drift detection inert | ✓ | | |
| 12. Challenger stub - no-op safety | | ✓ | |

### Key Insight: HITL Gates as the Nexus

All three research themes converge at the HITL gates:
- **Emergent behaviours** can silently degrade the information available at gates (Issues 1, 4, 8, 9, 11)
- **Safety constraints** are enforced *through* gates, but only if routing is correct and safety mechanisms are operational (Issues 2, 6, 10, 12)
- **Explainability** determines whether the user can *meaningfully* exercise their gate authority (Issues 3, 5, 7)

A gate that presents insufficient information, routes to the wrong recovery path, or requires unnecessary human intervention is a gate that undermines the responsible deployment posture of the entire system.

### Key Insight: Silent Failures as the Common Failure Mode

Issues 8, 9, 11, and 12 share a critical property: the system *appeared to function correctly* while silently operating at degraded capability. No error was raised, no warning was shown, and the output was delivered - but critical features (depth tracing, reference resolution, drift detection, adversarial review) were non-functional. This pattern represents the most dangerous class of deployment failure: one that passes all surface-level checks while providing none of the promised safety guarantees.

---

## References

- Arrieta, A.B. et al. (2020) 'Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI', *Information Fusion*, 58, pp. 82-115.
- Cemri, M. et al. (2025) 'Why do multi-agent LLM systems fail?', *NeurIPS 2025 Datasets and Benchmarks Track*.
- Guidotti, R. et al. (2018) 'A survey of methods for explaining black box models', *ACM Computing Surveys*, 51(5), pp. 1-42.
- Irving, G., Christiano, P. & Amodei, D. (2018) 'AI safety via debate', arXiv:1805.00899.
- Mosqueira-Rey, E. et al. (2023) 'Human-in-the-loop machine learning: a state of the art', *Artificial Intelligence Review*, 56(4), pp. 3005-3054.
