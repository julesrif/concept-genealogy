"""
api.py - FastAPI application exposing the Concept Genealogy pipeline as a web API.

The pipeline has three HITL (Human-in-the-Loop) gates that require user interaction.
To support this over HTTP, the pipeline is run in a background thread and pauses
at each gate, waiting for the user to respond via the API.

Endpoints:
  POST /api/runs            - Start a new pipeline run
  GET  /api/runs            - List all runs
  GET  /api/runs/{run_id}   - Get run status and current gate info
  POST /api/runs/{run_id}/gate - Respond to the current HITL gate
  GET  /api/runs/{run_id}/logs - Get pipeline log messages for a run
  GET  /api/runs/{run_id}/result - Get final result (JSON output)
"""

import logging
import os
import threading
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.config import load_config
from core.state import PipelineState, PipelineStage, CandidatePaper

# ── Fix SSL issues for HuggingFace in corporate environments ────────────────
# If a custom CA bundle exists (e.g. corporate proxy cert), use it for all
# HTTPS connections. Otherwise fall back to disabling SSL verification.
_CA_BUNDLE = Path(__file__).parent / "temp" / "BASF_internal_and_public_ca_bundle.crt"
if _CA_BUNDLE.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_CA_BUNDLE))
    os.environ.setdefault("CURL_CA_BUNDLE", str(_CA_BUNDLE))
    os.environ.setdefault("SSL_CERT_FILE", str(_CA_BUNDLE))
else:
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

logger = logging.getLogger(__name__)


# ── Run store (in-memory for MVP; could be Redis/DB for production) ─────────
class GateType(str, Enum):
    SEED_SELECTION = "seed_selection"
    CONCEPT_SELECTION = "concept_selection"
    GRAPH_REVIEW = "graph_review"
    SYNTHESIS_APPROVAL = "synthesis_approval"
    FINAL_APPROVAL = "final_approval"


class GateRequest(BaseModel):
    """Data presented to the user at an HITL gate."""
    gate_type: GateType
    candidates: list[dict] = Field(default_factory=list)  # Gate 1
    concepts: list[str] = Field(default_factory=list)      # DOI mode concept selection
    synthesis_text: Optional[str] = None                   # Gate 2 & 3
    challenger_critique: Optional[str] = None              # Gate 3


class RunStatus(str, Enum):
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"


class RunInfo(BaseModel):
    run_id: str
    user_concept: str
    status: RunStatus
    stage: str
    created_at: str
    current_gate: Optional[GateRequest] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    log_count: int = 0  # number of log entries available


class LogEntry(BaseModel):
    timestamp: str
    level: str
    source: str
    message: str


class RunStore:
    """Thread-safe store for pipeline run state and gate synchronisation."""

    MAX_LOGS_PER_RUN = 200

    def __init__(self):
        self._runs: dict[str, RunInfo] = {}
        self._gate_events: dict[str, threading.Event] = {}
        self._gate_responses: dict[str, dict] = {}
        self._pipeline_states: dict[str, PipelineState] = {}
        self._tracing_results: dict[str, list] = {}
        self._seed_papers: dict[str, dict] = {}
        self._concept_at_depth: dict[str, dict] = {}
        self._logs: dict[str, deque[LogEntry]] = {}
        self._lock = threading.Lock()

    def create_run(self, run_id: str, concept: str) -> RunInfo:
        info = RunInfo(
            run_id=run_id,
            user_concept=concept,
            status=RunStatus.RUNNING,
            stage=PipelineStage.INIT.value,
            created_at=datetime.now().isoformat(),
        )
        with self._lock:
            self._runs[run_id] = info
            self._gate_events[run_id] = threading.Event()
            self._logs[run_id] = deque(maxlen=self.MAX_LOGS_PER_RUN)
        return info

    def get_run(self, run_id: str) -> Optional[RunInfo]:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[RunInfo]:
        with self._lock:
            return list(self._runs.values())

    def set_gate(self, run_id: str, gate: GateRequest):
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].status = RunStatus.AWAITING_INPUT
                self._runs[run_id].current_gate = gate
                self._gate_events[run_id] = threading.Event()

    def respond_to_gate(self, run_id: str, response: dict):
        with self._lock:
            self._gate_responses[run_id] = response
            if run_id in self._runs:
                self._runs[run_id].status = RunStatus.RUNNING
                self._runs[run_id].current_gate = None
            if run_id in self._gate_events:
                self._gate_events[run_id].set()

    def wait_for_gate_response(self, run_id: str, timeout: float = 3600) -> Optional[dict]:
        event = self._gate_events.get(run_id)
        if event:
            event.wait(timeout=timeout)
            with self._lock:
                return self._gate_responses.pop(run_id, None)
        return None

    def set_completed(self, run_id: str, output_path: Optional[str] = None):
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].status = RunStatus.COMPLETED
                self._runs[run_id].stage = PipelineStage.DONE.value
                self._runs[run_id].output_path = output_path

    def set_failed(self, run_id: str, error: str):
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].status = RunStatus.FAILED
                self._runs[run_id].stage = PipelineStage.FAILED.value
                self._runs[run_id].error_message = error

    def update_stage(self, run_id: str, stage: str):
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].stage = stage

    def add_log(self, run_id: str, level: str, source: str, message: str):
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            source=source,
            message=message,
        )
        with self._lock:
            if run_id in self._logs:
                self._logs[run_id].append(entry)
            if run_id in self._runs:
                self._runs[run_id].log_count = len(self._logs.get(run_id, []))

    def get_logs(self, run_id: str, since: int = 0) -> list[LogEntry]:
        with self._lock:
            logs = self._logs.get(run_id, deque())
            return list(logs)[since:]

    def set_tracing_results(self, run_id: str, results: list, seed_paper: dict, concept_at_depth: dict = None):
        with self._lock:
            self._tracing_results[run_id] = results
            self._seed_papers[run_id] = seed_paper
            self._concept_at_depth[run_id] = concept_at_depth or {}

    def get_tracing_results(self, run_id: str) -> tuple[list, dict | None, dict]:
        with self._lock:
            return (
                self._tracing_results.get(run_id, []),
                self._seed_papers.get(run_id),
                self._concept_at_depth.get(run_id, {}),
            )


run_store = RunStore()


# ── Web-compatible HITL Gate ────────────────────────────────────────────────
class WebHITLGate:
    """
    HITL gate implementation that pauses the pipeline thread and waits
    for responses from the web API instead of stdin.
    """

    def __init__(self, run_id: str, store: RunStore):
        self.run_id = run_id
        self.store = store

    def gate_1_seed_selection(
        self, candidates: list[CandidatePaper]
    ) -> tuple[bool, Optional[CandidatePaper]]:
        gate = GateRequest(
            gate_type=GateType.SEED_SELECTION,
            candidates=[c.dict() for c in candidates],
        )
        self.store.set_gate(self.run_id, gate)
        response = self.store.wait_for_gate_response(self.run_id)

        if not response:
            return False, None

        action = response.get("action", "reject")
        if action == "approve":
            idx = response.get("selected_index", 0)
            if 0 <= idx < len(candidates):
                return True, candidates[idx]
        return False, None

    def gate_concept_selection(self, concepts: list[str]) -> list[str] | None:
        """Present extracted concepts for user selection (DOI mode)."""
        gate = GateRequest(
            gate_type=GateType.CONCEPT_SELECTION,
            concepts=concepts,
        )
        self.store.set_gate(self.run_id, gate)
        response = self.store.wait_for_gate_response(self.run_id)

        if not response:
            return None

        action = response.get("action", "reject")
        if action == "approve":
            indices = response.get("selected_indices", [])
            selected = []
            for idx in indices[:3]:
                if 0 <= idx < len(concepts):
                    selected.append(concepts[idx])
            return selected if selected else None
        return None

    def gate_2_synthesis_approval(self, synthesis_text: Optional[str]) -> bool:
        gate = GateRequest(
            gate_type=GateType.SYNTHESIS_APPROVAL,
            synthesis_text=synthesis_text,
        )
        self.store.set_gate(self.run_id, gate)
        response = self.store.wait_for_gate_response(self.run_id)

        if not response:
            return False
        return response.get("action") == "approve"

    def gate_graph_review(self) -> bool:
        """HITL gate for graph review before synthesis."""
        gate = GateRequest(gate_type=GateType.GRAPH_REVIEW)
        self.store.set_gate(self.run_id, gate)
        response = self.store.wait_for_gate_response(self.run_id)

        if not response:
            return False
        return response.get("action") == "approve"

    def gate_3_final_approval(
        self,
        synthesis_text: Optional[str],
        challenger_critique: Optional[str],
    ) -> bool:
        gate = GateRequest(
            gate_type=GateType.FINAL_APPROVAL,
            synthesis_text=synthesis_text,
            challenger_critique=challenger_critique,
        )
        self.store.set_gate(self.run_id, gate)
        response = self.store.wait_for_gate_response(self.run_id)

        if not response:
            return False
        return response.get("action") == "approve"


# ── Pipeline runner (background thread) ─────────────────────────────────────
class RunLogHandler(logging.Handler):
    """Captures log records into the RunStore for a specific run."""

    def __init__(self, run_id: str, store: RunStore):
        super().__init__()
        self.run_id = run_id
        self.store = store

    def emit(self, record: logging.LogRecord):
        try:
            self.store.add_log(
                self.run_id,
                level=record.levelname,
                source=record.name,
                message=self.format(record),
            )
        except Exception:
            pass


def run_pipeline_thread(run_id: str, concept: str, doi: str | None, config_dict: dict):
    """Execute the pipeline in a background thread with web HITL gates."""
    from core.orchestrator import Orchestrator

    # Inject a progress callback for user-facing messages only
    def emit_progress(message: str, level: str = "INFO"):
        run_store.add_log(run_id, level, "pipeline", message)

    config_dict["_progress_callback"] = emit_progress

    def on_tracing_complete(tracing_results, seed_paper, concept_at_depth=None):
        """Called by orchestrator when tracing is done, before synthesis."""
        results_dicts = [r.dict() for r in tracing_results]
        seed_dict = seed_paper.dict() if seed_paper else None
        run_store.set_tracing_results(run_id, results_dicts, seed_dict, concept_at_depth)

    config_dict["_tracing_complete_callback"] = on_tracing_complete

    display = f"DOI: {doi}" if doi else f"concept: \"{concept}\""
    emit_progress(f"Starting pipeline for {display}")

    try:
        # Create orchestrator with web HITL gate
        orchestrator = Orchestrator(config_dict)
        orchestrator.hitl = WebHITLGate(run_id, run_store)

        # Hook into stage updates
        original_wrap = orchestrator._wrap_node

        def stage_tracking_wrap(fn):
            wrapped = original_wrap(fn)

            def with_tracking(state: dict):
                stage = state.get("stage", "unknown")
                run_store.update_stage(run_id, stage)
                return wrapped(state)
            return with_tracking

        orchestrator._wrap_node = stage_tracking_wrap
        orchestrator.graph = orchestrator._build_graph()

        final_state = orchestrator.run(concept, input_doi=doi)

        if final_state.stage == PipelineStage.DONE:
            emit_progress("Pipeline completed successfully.")
            run_store.set_completed(run_id, final_state.output_path)
        else:
            error_msg = final_state.error_message or "Unknown error"
            emit_progress(f"Pipeline failed: {error_msg}", "ERROR")
            run_store.set_failed(run_id, error_msg)

    except Exception as e:
        logger.exception("Pipeline run %s failed", run_id)
        emit_progress(f"An error occurred: {e}", "ERROR")
        run_store.set_failed(run_id, str(e))


# ── FastAPI app ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Concept Genealogy API starting")
    yield
    logger.info("Concept Genealogy API shutting down")


app = FastAPI(
    title="Concept Genealogy API",
    description="LLM-powered academic concept tracing and genealogy",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Models ──────────────────────────────────────────────────────────────
class StartRunRequest(BaseModel):
    concept: Optional[str] = None
    doi: Optional[str] = None
    depth: Optional[int] = None
    papers_per_level: Optional[int] = None


class GateResponse(BaseModel):
    action: str  # "approve" or "reject"
    selected_index: Optional[int] = None  # For gate 1 only
    selected_indices: Optional[list[int]] = None  # For concept selection gate


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.post("/api/runs", response_model=RunInfo)
async def start_run(req: StartRunRequest):
    """Start a new concept genealogy pipeline run."""
    # Validate: either concept or doi must be provided
    if not req.concept and not req.doi:
        raise HTTPException(status_code=400, detail="Either 'concept' or 'doi' must be provided.")
    if req.concept and req.doi:
        raise HTTPException(status_code=400, detail="Provide either 'concept' or 'doi', not both.")

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    config = load_config()
    config_dict = config.dict()
    if req.depth is not None:
        config_dict["max_depth"] = req.depth
    if req.papers_per_level is not None:
        config_dict["papers_per_level"] = req.papers_per_level

    display_name = req.concept if req.concept else f"DOI: {req.doi}"
    info = run_store.create_run(run_id, display_name)

    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(run_id, req.concept or "", req.doi, config_dict),
        daemon=True,
    )
    thread.start()

    return info


@app.get("/api/runs", response_model=list[RunInfo])
async def list_runs():
    """List all pipeline runs."""
    return run_store.list_runs()


@app.get("/api/runs/{run_id}", response_model=RunInfo)
async def get_run(run_id: str):
    """Get current status and gate info for a run."""
    info = run_store.get_run(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    return info


@app.post("/api/runs/{run_id}/gate")
async def respond_to_gate(run_id: str, response: GateResponse):
    """Respond to the current HITL gate for a run."""
    info = run_store.get_run(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info.status != RunStatus.AWAITING_INPUT:
        raise HTTPException(status_code=400, detail="Run is not awaiting input")

    run_store.respond_to_gate(run_id, response.dict())
    return {"status": "ok"}


@app.get("/api/runs/{run_id}/logs", response_model=list[LogEntry])
async def get_logs(run_id: str, since: int = 0):
    """Get log messages for a run. Use ?since=N to get entries after index N."""
    info = run_store.get_run(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_store.get_logs(run_id, since=since)


@app.get("/api/runs/{run_id}/graph")
async def get_graph(run_id: str):
    """
    Build a concept genealogy graph from the tracing results.

    Returns nodes (papers at each depth) and edges showing the actual citation
    chain — which paper's references led to which traced paper. This accurately
    represents the BFS traversal the pipeline performed.
    """
    info = run_store.get_run(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")

    tracing_results, seed_paper, concept_at_depth = run_store.get_tracing_results(run_id)
    if not tracing_results:
        return {"nodes": [], "edges": [], "concept_at_depth": {}}

    nodes = []
    edges = []
    paper_ids_added: set[str] = set()

    # Build a set of all traced paper IDs for edge resolution
    traced_paper_ids = {r["paper_id"] for r in tracing_results}

    # Add seed paper as root node
    if seed_paper:
        seed_id = seed_paper["paper_id"]
        nodes.append({
            "id": seed_id,
            "type": "seed",
            "label": seed_paper.get("title", "Seed Paper"),
            "authors": seed_paper.get("authors", []),
            "doi": seed_paper.get("doi"),
            "depth": 0,
            "year": seed_paper.get("year"),
            "claims_count": 0,
        })
        paper_ids_added.add(seed_id)

    # Process each tracing result — add paper nodes
    for result in tracing_results:
        paper_id = result["paper_id"]
        depth = result.get("depth", 0)
        claims = result.get("claims", [])

        if paper_id not in paper_ids_added:
            nodes.append({
                "id": paper_id,
                "type": "seed" if depth == 0 else "reference",
                "label": result.get("paper_title", paper_id),
                "authors": result.get("authors", []),
                "doi": result.get("doi"),
                "depth": depth,
                "year": result.get("year"),
                "claims_count": len(claims),
            })
            paper_ids_added.add(paper_id)

        # Update claims count for seed paper
        if depth == 0 and seed_paper and paper_id == seed_paper["paper_id"]:
            for node in nodes:
                if node["id"] == paper_id:
                    node["claims_count"] = len(claims)

    # Build edges from the citation chain: if paper A at depth N has
    # paper B in its references_explored, and paper B was traced at depth N+1,
    # then add edge A -> B
    # Map paper_id -> depth for concept labelling
    paper_depth_map = {r["paper_id"]: r.get("depth", 0) for r in tracing_results}

    for result in tracing_results:
        parent_id = result["paper_id"]
        parent_depth = result.get("depth", 0)
        refs_explored = result.get("references_explored", [])

        for ref_id in refs_explored:
            if ref_id in traced_paper_ids and ref_id in paper_ids_added:
                # Edge concept = the concept active at the target's depth
                target_depth = paper_depth_map.get(ref_id, parent_depth + 1)
                edge_concept = concept_at_depth.get(str(target_depth)) or concept_at_depth.get(target_depth) or info.user_concept
                edges.append({
                    "source": parent_id,
                    "target": ref_id,
                    "concept": edge_concept,
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "concept_at_depth": concept_at_depth,
    }


@app.get("/api/runs/{run_id}/result")
async def get_result(run_id: str):
    """Get the final JSON result for a completed run."""
    info = run_store.get_run(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="Run not found")
    if info.status != RunStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Run not yet completed")

    if info.output_path:
        json_path = Path(info.output_path)
        if json_path.exists() and json_path.suffix == ".json":
            import json
            return json.loads(json_path.read_text(encoding="utf-8"))
        # Try to find matching JSON file
        md_path = Path(info.output_path)
        json_path = md_path.with_suffix(".json")
        if json_path.exists():
            import json
            return json.loads(json_path.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail="Result file not found")


@app.get("/api/outputs")
async def list_outputs():
    """List all available output files."""
    output_dir = Path("output")
    if not output_dir.exists():
        return []

    results = []
    for f in sorted(output_dir.glob("*.json"), reverse=True):
        import json
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "filename": f.name,
                "run_id": data.get("run_id", f.stem),
                "user_concept": data.get("user_concept", "unknown"),
                "seed_paper_title": data.get("seed_paper", {}).get("title", ""),
                "created_at": data.get("started_at", ""),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


@app.get("/api/outputs/{filename}")
async def get_output(filename: str):
    """Get a specific output file by filename."""
    import json
    # Prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    filepath = Path("output") / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return json.loads(filepath.read_text(encoding="utf-8"))
