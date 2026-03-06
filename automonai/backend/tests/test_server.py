"""Tests for the FastAPI app assembly in automonai/backend/server.py."""

import pytest
from fastapi.testclient import TestClient

from automonai.backend.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestRootEndpoint:
    """Tests for the GET / endpoint serving index.html."""

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]
        assert "<!DOCTYPE" in resp.text or "<html" in resp.text


class TestRoutersMounted:
    """Tests that API routers are correctly mounted on the app."""

    def test_config_router_mounted(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_configs_router_mounted(self, client):
        resp = client.get("/api/configs/list")
        assert resp.status_code == 200


class TestStaticFiles:
    """Tests that static files are served from the frontend directory."""

    def test_static_js_accessible(self, client):
        resp = client.get("/static/js/init.js")
        assert resp.status_code == 200

    def test_static_missing_file_returns_404(self, client):
        resp = client.get("/static/nonexistent.xyz")
        assert resp.status_code == 404
