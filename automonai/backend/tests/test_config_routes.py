"""Tests for the config API routes (models, datasets, options)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from automonai.backend.routers.config import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestGetModels:
    """Tests for GET /api/models."""

    def test_returns_200(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_returns_dict(self, client):
        resp = client.get("/api/models")
        assert isinstance(resp.json(), dict)

    @pytest.mark.parametrize(
        "model_key",
        ["unet", "attention_unet", "segresnet", "swinunetr"],
    )
    def test_contains_expected_models(self, client, model_key):
        models = client.get("/api/models").json()
        assert model_key in models

    def test_each_model_has_name_and_description(self, client):
        models = client.get("/api/models").json()
        for key, value in models.items():
            assert "name" in value, f"Model {key} missing 'name'"
            assert "description" in value, f"Model {key} missing 'description'"


class TestGetDatasets:
    """Tests for GET /api/datasets."""

    def test_returns_200(self, client):
        resp = client.get("/api/datasets")
        assert resp.status_code == 200

    def test_returns_dict(self, client):
        resp = client.get("/api/datasets")
        assert isinstance(resp.json(), dict)


class TestGetOptions:
    """Tests for GET /api/options."""

    def test_returns_200(self, client):
        resp = client.get("/api/options")
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        "key",
        [
            "losses",
            "metrics",
            "optimizers",
            "schedulers",
            "inferers",
            "augmentation_transforms",
            "dataset_classes",
        ],
    )
    def test_has_expected_key(self, client, key):
        data = client.get("/api/options").json()
        assert key in data

    @pytest.mark.parametrize("loss", ["dice", "cross_entropy", "focal", "dice_ce"])
    def test_losses_contains_expected_items(self, client, loss):
        losses = client.get("/api/options").json()["losses"]
        assert isinstance(losses, list)
        assert loss in losses

    @pytest.mark.parametrize("metric", ["dice", "iou", "hausdorff"])
    def test_metrics_contains_expected_items(self, client, metric):
        metrics = client.get("/api/options").json()["metrics"]
        assert isinstance(metrics, list)
        assert metric in metrics
