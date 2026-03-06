"""Tests for the launch API routes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from automonai.backend.routers.launch import router, _processes, _lock


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_processes():
    """Clear process registry between tests."""
    with _lock:
        _processes.clear()
    yield
    with _lock:
        _processes.clear()


class TestLaunchValidation:
    """Tests for POST /api/launch command validation."""

    def test_invalid_command_returns_400(self, client):
        resp = client.post("/api/launch", json={"command": "rm -rf /"})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid command"

    def test_partial_prefix_returns_400(self, client):
        resp = client.post("/api/launch", json={"command": "python3 -m automonai.core"})
        assert resp.status_code == 400

    def test_empty_command_returns_400(self, client):
        resp = client.post("/api/launch", json={"command": ""})
        assert resp.status_code == 400


class TestLaunchStatus:
    """Tests for GET /api/launch/status."""

    def test_nonexistent_run_returns_not_running(self, client):
        resp = client.get("/api/launch/status", params={"run_id": "nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["returncode"] is None

    def test_default_run_id_returns_not_running(self, client):
        resp = client.get("/api/launch/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["returncode"] is None


class TestLaunchStop:
    """Tests for POST /api/launch/stop idempotency."""

    def test_stop_nonexistent_run_returns_stopped(self, client):
        resp = client.post("/api/launch/stop", json={"run_id": "nonexistent"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_stop_default_run_returns_stopped(self, client):
        resp = client.post("/api/launch/stop", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_stop_no_body_returns_stopped(self, client):
        resp = client.post("/api/launch/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"


class TestLaunchList:
    """Tests for GET /api/launch/list."""

    def test_empty_list_returns_empty_dict(self, client):
        resp = client.get("/api/launch/list")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_list_includes_variant_qualified_run_ids(self, client):
        class DummyProc:
            def poll(self):
                return None

        with _lock:
            _processes["cfg__fold_3"] = {
                "proc": DummyProc(),
                "log_buffer": [],
                "finished_at": None,
            }
            _processes["cfg__fold_4"] = {
                "proc": DummyProc(),
                "log_buffer": [],
                "finished_at": None,
            }

        resp = client.get("/api/launch/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "cfg__fold_3" in data
        assert "cfg__fold_4" in data
        assert data["cfg__fold_3"]["running"] is True
