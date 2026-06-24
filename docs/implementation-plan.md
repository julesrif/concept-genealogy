Jira Board [Link](https://intelligent-agents.atlassian.net/jira/software/projects/KAN/boards/1)

# Week 1 ✅
Infrastructure + Orchestrator skeleton
- [x] Repo structure - monorepo, Docker Compose (PostgreSQL + pgvector), .env template
- [x] Orchestrator base - LangGraph StateGraph, global state schema (Pydantic), routing logic
- [x] Retriever agent - Semantic Scholar API wrapper, SJR CSV ranking, top-3 candidate selection
- [x] HITL Gate 1 - CLI prompt for seed approval
- [x] pytest unit tests for Retriever + state schema (13 + 20 tests)

# Week 2 ✅
Citation Extractor + Evaluator + Inspector
- [x] Citation Extractor - PyMuPDF PDF download, claim extraction prompt (Groq llama-3.1-8b), structured output
- [x] Evaluator - concept filtering, claim validity scoring, drift detection via embeddings
- [x] Inspector - Pydantic schema validation at every agent boundary, max-1-retry logic
- [x] Recursive loop - depth cap, seed tracking, SJR-bounded traversal, concept abstraction
- [x] pytest for extractor + validation + golden fixture test (11 + 9 + 5 tests)

# Week 3 ✅
Synthesiser + Challenger + HITL + storage + stretch
- [x] Synthesiser - grounded summary via llama-3.3-70b, Markdown output
- [x] Challenger - single-sided adversarial review (replaced stub with full implementation)
- [x] Storage - PostgreSQL persistence, JSON + Markdown dual output, pgvector embeddings
- [x] HITL Gates - 4 web-based gates (seed selection, graph review, synthesis approval, final approval)
- [x] pgvector embeddings (all-mpnet-base-v2, 768d) - drift detection operational
- [x] End-to-end functional test with "attention mechanism"
- [x] CrossRef reference fallback + Elsevier full-text API
- [x] DOI input mode with concept extraction
- [x] Depth-adaptive concept abstraction

# Week 4 (current)
Documentation + testing evidence + presentation
- [x] README - purpose, architecture (Mermaid diagrams), install + run, sources, deviations table
- [x] Technology Stack & Acknowledgments - all libraries/models/APIs with licenses
- [x] Code comments - all backend agents have "why" docstrings with Harvard citations
- [x] Code comments - all Vue components have design-decision header comments
- [x] Test evidence - pytest output saved (129 tests passing, `backend/test-output/test-results.txt`)
- [x] Coverage report saved (63% overall, `backend/test-output/coverage-report.txt`)
- [x] Frontend - Vue 3 SPA with live pipeline monitoring, interactive HITL gates, graph visualisation
- [ ] PowerPoint deck - 10 slides max, covers all grading criteria
- [ ] Audio transcript - narration script aligned to slides
- [ ] Screenshots of execution (optional - test evidence + output files may suffice)

# Pre-submission checklist
- [ ] Verify no `.env` or secrets in git history (`git log --all --diff-filter=A -- '*.env'`)
- [ ] Verify commit messages are descriptive (no "WIP" / "fix" without context)
- [ ] Verify no large binaries in history
- [ ] Confirm `requirements.txt` versions are appropriate
- [ ] Final `git push` to submission branch