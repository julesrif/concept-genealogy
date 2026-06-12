Jira Board [Link](https://intelligent-agents.atlassian.net/jira/software/projects/KAN/boards/1)

# Week 1
Infrastructure + Orchestrator skeleton
Repo structure - monorepo, Docker Compose (PostgreSQL + pgvector), .env template code
Orchestrator base - LangGraph StateGraph, global state schema (Pydantic), routing logic stub code
Retriever agent - Semantic Scholar API wrapper, SJR CSV ranking, top-3 candidate selection code
HITL Gate 1 - CLI prompt for seed approval (simple, robust) code
pytest unit tests for Retriever + state schema testing

# Week 2
Citation Extractor + Evaluator + Inspector
Citation Extractor - PyMuPDF PDF download, claim extraction prompt (Groq llama-3.1-8b), structured output code
Evaluator - concept filtering, claim validity scoring, vector-ready JSON prep code
Inspector - Pydantic schema validation at every agent boundary, max-1-retry logic code
Recursive loop - depth cap, seed tracking, SJR-bounded traversal code
pytest for extractor + validation; golden fixture test with a known paper testing

# Week 3
Synthesiser + HITL 2 & 3 + storage + stretch
Synthesiser - grounded summary via llama-3.3-70b, Markdown output, HITL Gate 2 code
Challenger stub - placeholder node in LangGraph with documented reason for deferral code doc
Storage - PostgreSQL persistence, JSON log output, HITL Gate 3 + final delivery code
Stretch: pgvector embeddings (specter2 / all-mpnet) if time allows code
End-to-end functional test with a real concept (e.g. "attention mechanism") testing

# Week 4
Documentation + testing evidence + presentation
README - purpose, architecture overview, install + run instructions, sources doc
Code comments - pass through all agents, ensure every "why" decision is cited doc
Test evidence - capture pytest output, screenshots of execution, log captures testing
PowerPoint deck - 10 slides max, covers all grading criteria pres
Audio transcript - narration script aligned to slides pres
Final GitHub push: clean history, no .env committed, requirements.txt pinned doc

