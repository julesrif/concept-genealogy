"""
tests/test_api.py - Unit tests for the FastAPI API layer.

Tests:
  - POST /api/runs: start with concept, doi, both (error), neither (error)
  - GET /api/runs: list runs
  - GET /api/runs/{run_id}: get specific run
  - POST /api/runs/{run_id}/gate: respond to HITL gate
  - RunStore thread-safe operations
  - WebHITLGate behaviour

Implementation plan coverage: Week 3 (API integration tests)
"""

import threading
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api import app, run_store, RunStore, RunStatus, GateType, GateRequest, WebHITLGate


@pytest.fixture(autouse=True)
def clean_run_store():
    """Reset the global run store before each test."""
    run_store._runs.clear()
    run_store._gate_events.clear()
    run_store._gate_responses.clear()
    run_store._pipeline_states.clear()
    run_store._tracing_results.clear()
    run_store._seed_papers.clear()
    run_store._concept_at_depth.clear()
    run_store._logs.clear()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient (no network, no background thread for pipeline)."""
    return TestClient(app)


class TestStartRun:
    """Tests for POST /api/runs."""

    @patch("api.threading.Thread")
    def test_start_with_concept(self, mock_thread, client):
        """Starting a run with a concept returns 200 and run info."""
        mock_thread.return_value = MagicMock()
        resp = client.post("/api/runs", json={"concept": "attention mechanism"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_concept"] == "attention mechanism"
        assert data["status"] == "running"
        assert data["run_id"].startswith("run_")

    @patch("api.threading.Thread")
    def test_start_with_doi(self, mock_thread, client):
        """Starting a run with a DOI returns 200."""
        mock_thread.return_value = MagicMock()
        resp = client.post("/api/runs", json={"doi": "10.48550/arXiv.1706.03762"})
        assert resp.status_code == 200
        data = resp.json()
        assert "DOI:" in data["user_concept"]

    @patch("api.threading.Thread")
    def test_start_with_both_concept_and_doi_fails(self, mock_thread, client):
        """Providing both concept and doi returns 400."""
        resp = client.post("/api/runs", json={
            "concept": "attention", "doi": "10.1234/test"
        })
        assert resp.status_code == 400
        assert "not both" in resp.json()["detail"]

    def test_start_with_neither_fails(self, client):
        """Providing neither concept nor doi returns 400."""
        resp = client.post("/api/runs", json={})
        assert resp.status_code == 400
        assert "must be provided" in resp.json()["detail"]


class TestListRuns:
    """Tests for GET /api/runs."""

    def test_empty_list(self, client):
        """No runs returns empty array."""
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created_runs(self, client):
        """Created runs appear in list."""
        run_store.create_run("test_run_1", "concept A")
        run_store.create_run("test_run_2", "concept B")
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestGetRun:
    """Tests for GET /api/runs/{run_id}."""

    def test_existing_run(self, client):
        """Returns run info for a valid run_id."""
        run_store.create_run("test_run_1", "attention mechanism")
        resp = client.get("/api/runs/test_run_1")
        assert resp.status_code == 200
        assert resp.json()["user_concept"] == "attention mechanism"

    def test_missing_run_404(self, client):
        """Returns 404 for unknown run_id."""
        resp = client.get("/api/runs/nonexistent")
        assert resp.status_code == 404


class TestGateResponse:
    """Tests for POST /api/runs/{run_id}/gate."""

    def test_respond_while_awaiting(self, client):
        """Can respond to gate when run is awaiting input."""
        run_store.create_run("test_run", "concept")
        run_store.set_gate("test_run", GateRequest(gate_type=GateType.SEED_SELECTION))
        resp = client.post("/api/runs/test_run/gate", json={
            "action": "approve", "selected_index": 0
        })
        assert resp.status_code == 200

    def test_respond_when_not_awaiting_fails(self, client):
        """Responding to gate when not awaiting returns 400."""
        run_store.create_run("test_run", "concept")
        resp = client.post("/api/runs/test_run/gate", json={"action": "approve"})
        assert resp.status_code == 400

    def test_respond_to_missing_run_404(self, client):
        """Responding to unknown run_id returns 404."""
        resp = client.post("/api/runs/nonexistent/gate", json={"action": "approve"})
        assert resp.status_code == 404


class TestRunLogs:
    """Tests for GET /api/runs/{run_id}/logs."""

    def test_logs_initially_empty(self, client):
        """New run has no logs."""
        run_store.create_run("test_run", "concept")
        resp = client.get("/api/runs/test_run/logs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_logs_accumulate(self, client):
        """Added logs are returned."""
        run_store.create_run("test_run", "concept")
        run_store.add_log("test_run", "INFO", "retriever", "Searching S2...")
        run_store.add_log("test_run", "INFO", "retriever", "Found 5 papers")
        resp = client.get("/api/runs/test_run/logs")
        assert len(resp.json()) == 2

    def test_since_parameter(self, client):
        """since=N returns only logs after index N."""
        run_store.create_run("test_run", "concept")
        for i in range(5):
            run_store.add_log("test_run", "INFO", "test", f"Log {i}")
        resp = client.get("/api/runs/test_run/logs?since=3")
        logs = resp.json()
        assert len(logs) == 2
        assert "Log 3" in logs[0]["message"]


class TestRunStore:
    """Direct unit tests for RunStore (thread safety, state transitions)."""

    def test_create_and_get(self):
        store = RunStore()
        store.create_run("r1", "concept")
        info = store.get_run("r1")
        assert info is not None
        assert info.status == RunStatus.RUNNING

    def test_set_completed(self):
        store = RunStore()
        store.create_run("r1", "concept")
        store.set_completed("r1", "/output/result.json")
        info = store.get_run("r1")
        assert info.status == RunStatus.COMPLETED
        assert info.output_path == "/output/result.json"

    def test_set_failed(self):
        store = RunStore()
        store.create_run("r1", "concept")
        store.set_failed("r1", "Something went wrong")
        info = store.get_run("r1")
        assert info.status == RunStatus.FAILED
        assert info.error_message == "Something went wrong"

    def test_gate_synchronisation(self):
        """Gate event allows cross-thread synchronisation."""
        store = RunStore()
        store.create_run("r1", "concept")
        store.set_gate("r1", GateRequest(gate_type=GateType.SEED_SELECTION))

        # Simulate a frontend response in another thread
        def respond():
            import time
            time.sleep(0.1)
            store.respond_to_gate("r1", {"action": "approve", "selected_index": 0})

        t = threading.Thread(target=respond)
        t.start()

        # This should unblock once the response arrives
        response = store.wait_for_gate_response("r1", timeout=5)
        t.join()

        assert response == {"action": "approve", "selected_index": 0}

    def test_max_logs_limit(self):
        """Logs are bounded by MAX_LOGS_PER_RUN."""
        store = RunStore()
        store.create_run("r1", "concept")
        for i in range(300):
            store.add_log("r1", "INFO", "test", f"Log {i}")
        logs = store.get_logs("r1")
        assert len(logs) == RunStore.MAX_LOGS_PER_RUN


class TestWebHITLGate:
    """Tests for WebHITLGate behaviour."""

    def test_gate_1_approve(self):
        """gate_1_seed_selection returns (True, selected paper) on approve."""
        from core.state import CandidatePaper

        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        candidates = [
            CandidatePaper(paper_id="p1", title="Paper 1", authors=[]),
            CandidatePaper(paper_id="p2", title="Paper 2", authors=[]),
        ]

        # Respond in background
        def respond():
            import time
            time.sleep(0.1)
            store.respond_to_gate("r1", {"action": "approve", "selected_index": 1})

        t = threading.Thread(target=respond)
        t.start()

        approved, paper = gate.gate_1_seed_selection(candidates)
        t.join()

        assert approved is True
        assert paper.paper_id == "p2"

    def test_gate_1_reject(self):
        """gate_1_seed_selection returns (False, None) on reject."""
        from core.state import CandidatePaper

        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        candidates = [CandidatePaper(paper_id="p1", title="P", authors=[])]

        def respond():
            import time
            time.sleep(0.1)
            store.respond_to_gate("r1", {"action": "reject"})

        t = threading.Thread(target=respond)
        t.start()

        approved, paper = gate.gate_1_seed_selection(candidates)
        t.join()

        assert approved is False
        assert paper is None

    def test_gate_1_no_response(self):
        """gate_1_seed_selection returns (False, None) when response is None."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        from core.state import CandidatePaper
        candidates = [CandidatePaper(paper_id="p1", title="P", authors=[])]

        # Respond with None (timeout simulation)
        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", None)

        t = threading.Thread(target=respond)
        t.start()
        approved, paper = gate.gate_1_seed_selection(candidates)
        t.join()
        # None response treated as reject
        assert approved is False

    def test_gate_1_invalid_index_rejects(self):
        """gate_1_seed_selection rejects when selected_index is out of range."""
        from core.state import CandidatePaper
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        candidates = [CandidatePaper(paper_id="p1", title="P", authors=[])]

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "approve", "selected_index": 99})

        t = threading.Thread(target=respond)
        t.start()
        approved, paper = gate.gate_1_seed_selection(candidates)
        t.join()
        assert approved is False
        assert paper is None

    def test_gate_concept_selection_approve(self):
        """gate_concept_selection returns selected concepts on approve."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        concepts = ["attention", "transformer", "self-attention", "encoder"]

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "approve", "selected_indices": [0, 2]})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_concept_selection(concepts)
        t.join()
        assert result == ["attention", "self-attention"]

    def test_gate_concept_selection_reject(self):
        """gate_concept_selection returns None on reject."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "reject"})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_concept_selection(["a", "b"])
        t.join()
        assert result is None

    def test_gate_concept_selection_no_response(self):
        """gate_concept_selection returns None when response is None."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", None)

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_concept_selection(["a"])
        t.join()
        assert result is None

    def test_gate_concept_selection_empty_indices(self):
        """gate_concept_selection with empty indices returns None."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "approve", "selected_indices": []})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_concept_selection(["a", "b"])
        t.join()
        assert result is None

    def test_gate_concept_selection_limits_to_3(self):
        """gate_concept_selection limits selection to 3 concepts."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        concepts = ["a", "b", "c", "d", "e"]

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "approve", "selected_indices": [0, 1, 2, 3, 4]})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_concept_selection(concepts)
        t.join()
        assert len(result) == 3

    def test_gate_2_synthesis_approve(self):
        """gate_2_synthesis_approval returns True on approve."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "approve"})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_2_synthesis_approval("Synthesis text")
        t.join()
        assert result is True

    def test_gate_2_synthesis_reject(self):
        """gate_2_synthesis_approval returns False on reject."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "reject"})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_2_synthesis_approval("Synthesis text")
        t.join()
        assert result is False

    def test_gate_2_no_response(self):
        """gate_2_synthesis_approval returns False when no response."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", None)

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_2_synthesis_approval("text")
        t.join()
        assert result is False

    def test_gate_graph_review_approve(self):
        """gate_graph_review returns True on approve."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "approve"})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_graph_review()
        t.join()
        assert result is True

    def test_gate_graph_review_reject(self):
        """gate_graph_review returns False on reject."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "reject"})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_graph_review()
        t.join()
        assert result is False

    def test_gate_3_final_approve(self):
        """gate_3_final_approval returns True on approve."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "approve"})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_3_final_approval("synthesis", "critique", [])
        t.join()
        assert result is True

    def test_gate_3_final_reject(self):
        """gate_3_final_approval returns False on reject."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", {"action": "reject"})

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_3_final_approval("synthesis", "critique", [])
        t.join()
        assert result is False

    def test_gate_3_no_response(self):
        """gate_3_final_approval returns False when no response."""
        store = RunStore()
        store.create_run("r1", "concept")
        gate = WebHITLGate("r1", store)

        def respond():
            import time
            time.sleep(0.05)
            store.respond_to_gate("r1", None)

        t = threading.Thread(target=respond)
        t.start()
        result = gate.gate_3_final_approval("s", "c")
        t.join()
        assert result is False


# ── Run pipeline thread ────────────────────────────────────────────────────

class TestRunPipelineThread:
    """Tests for run_pipeline_thread."""

    def test_successful_pipeline(self, client):
        """Pipeline thread sets completed on success."""
        from api import run_pipeline_thread, run_store
        from core.state import PipelineState, PipelineStage

        run_store.create_run("r1", "test concept")

        with patch("core.orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            MockOrch.return_value = mock_orch
            mock_orch.run.return_value = PipelineState(
                user_concept="test",
                stage=PipelineStage.DONE,
                output_path="output/test.json",
            )
            mock_orch._wrap_node = MagicMock(side_effect=lambda fn: fn)
            mock_orch._build_graph.return_value = mock_orch.graph

            run_pipeline_thread("r1", "test concept", None, {"groq_api_key": ""})

        info = run_store.get_run("r1")
        assert info.status == RunStatus.COMPLETED

    def test_failed_pipeline(self, client):
        """Pipeline thread sets failed on error state."""
        from api import run_pipeline_thread, run_store
        from core.state import PipelineState, PipelineStage

        run_store.create_run("r1", "test")

        with patch("core.orchestrator.Orchestrator") as MockOrch:
            mock_orch = MagicMock()
            MockOrch.return_value = mock_orch
            mock_orch.run.return_value = PipelineState(
                user_concept="test",
                stage=PipelineStage.FAILED,
                error_message="DOI not found",
            )
            mock_orch._wrap_node = MagicMock(side_effect=lambda fn: fn)
            mock_orch._build_graph.return_value = mock_orch.graph

            run_pipeline_thread("r1", "test", None, {"groq_api_key": ""})

        info = run_store.get_run("r1")
        assert info.status == RunStatus.FAILED
        assert "DOI not found" in info.error_message

    def test_exception_in_pipeline(self, client):
        """Pipeline thread sets failed when exception is raised."""
        from api import run_pipeline_thread, run_store

        run_store.create_run("r1", "test")

        with patch("core.orchestrator.Orchestrator") as MockOrch:
            MockOrch.side_effect = RuntimeError("Import failed")

            run_pipeline_thread("r1", "test", None, {"groq_api_key": ""})

        info = run_store.get_run("r1")
        assert info.status == RunStatus.FAILED


# ── Graph endpoint ──────────────────────────────────────────────────────────

class TestGraphEndpoint:
    """Tests for GET /api/runs/{run_id}/graph."""

    def test_graph_no_tracing_results(self, client):
        run_store.create_run("r1", "concept")
        resp = client.get("/api/runs/r1/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_graph_with_tracing_results(self, client):
        run_store.create_run("r1", "attention mechanism")
        seed = {"paper_id": "seed1", "title": "Seed Paper", "authors": ["A"], "doi": "10.1/s", "year": 2020}
        tracing = [
            {
                "paper_id": "seed1",
                "paper_title": "Seed Paper",
                "depth": 0,
                "claims": [{"claim_text": "claim1"}],
                "references_explored": ["ref1"],
                "authors": ["A"],
                "doi": "10.1/s",
                "year": 2020,
            },
            {
                "paper_id": "ref1",
                "paper_title": "Ref Paper",
                "depth": 1,
                "claims": [{"claim_text": "claim2"}],
                "references_explored": [],
                "authors": ["B"],
                "doi": "10.1/r",
                "year": 2018,
            },
        ]
        run_store.set_tracing_results("r1", tracing, seed, {0: "attention", 1: "selective attention"})

        resp = client.get("/api/runs/r1/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["edges"][0]["source"] == "seed1"
        assert data["edges"][0]["target"] == "ref1"

    def test_graph_missing_run_404(self, client):
        resp = client.get("/api/runs/nonexistent/graph")
        assert resp.status_code == 404


# ── Result endpoint ────────────────────────────────────────────────────────

class TestResultEndpoint:
    """Tests for GET /api/runs/{run_id}/result."""

    def test_result_not_completed_400(self, client):
        run_store.create_run("r1", "concept")
        resp = client.get("/api/runs/r1/result")
        assert resp.status_code == 400
        assert "not yet completed" in resp.json()["detail"]

    def test_result_no_output_path_404(self, client):
        run_store.create_run("r1", "concept")
        run_store.set_completed("r1", None)
        resp = client.get("/api/runs/r1/result")
        assert resp.status_code == 404

    def test_result_json_file(self, client, tmp_path):
        import json
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"concept": "test"}), encoding="utf-8")

        run_store.create_run("r1", "concept")
        run_store.set_completed("r1", str(result_file))
        resp = client.get("/api/runs/r1/result")
        assert resp.status_code == 200
        assert resp.json()["concept"] == "test"

    def test_result_md_path_tries_json_suffix(self, client, tmp_path):
        """When output_path is .md, tries .json suffix."""
        import json
        json_file = tmp_path / "result.json"
        json_file.write_text(json.dumps({"found": True}), encoding="utf-8")

        run_store.create_run("r1", "concept")
        run_store.set_completed("r1", str(tmp_path / "result.md"))
        resp = client.get("/api/runs/r1/result")
        assert resp.status_code == 200
        assert resp.json()["found"] is True

    def test_result_missing_run_404(self, client):
        resp = client.get("/api/runs/nonexistent/result")
        assert resp.status_code == 404


# ── Outputs endpoints ──────────────────────────────────────────────────────

class TestOutputsEndpoints:
    """Tests for /api/outputs and /api/outputs/{filename}."""

    def test_list_outputs_no_dir(self, client, tmp_path, monkeypatch):
        """Returns empty list when output dir doesn't exist."""
        monkeypatch.chdir(tmp_path)
        resp = client.get("/api/outputs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_outputs_with_files(self, client, tmp_path, monkeypatch):
        """Lists output JSON files."""
        import json
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "result.json").write_text(
            json.dumps({"user_concept": "test", "seed_paper": {"title": "T"}, "started_at": "2024-01-01"}),
            encoding="utf-8",
        )
        resp = client.get("/api/outputs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["user_concept"] == "test"

    def test_get_output_path_traversal_with_backslash(self, client):
        """Path traversal with backslash is rejected."""
        resp = client.get("/api/outputs/..%5Cetc%5Cpasswd")
        assert resp.status_code == 400
        assert "Invalid filename" in resp.json()["detail"]

    def test_get_output_missing_file(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "output").mkdir()
        resp = client.get("/api/outputs/nonexistent.json")
        assert resp.status_code == 404

    def test_get_output_success(self, client, tmp_path, monkeypatch):
        import json
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "result.json").write_text(
            json.dumps({"concept": "test"}), encoding="utf-8"
        )
        resp = client.get("/api/outputs/result.json")
        assert resp.status_code == 200
        assert resp.json()["concept"] == "test"


# ── RunLogHandler ──────────────────────────────────────────────────────────

class TestRunLogHandler:
    """Tests for RunLogHandler."""

    def test_emits_log_to_store(self):
        import logging
        from api import RunLogHandler

        store = RunStore()
        store.create_run("r1", "concept")
        handler = RunLogHandler("r1", store)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None,
        )
        handler.emit(record)

        logs = store.get_logs("r1")
        assert len(logs) == 1
        assert "Test message" in logs[0].message
