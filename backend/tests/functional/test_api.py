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
