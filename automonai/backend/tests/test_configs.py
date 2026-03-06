"""Tests for the configs API routes."""

import json
import shutil

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from automonai.backend.routers.configs import router, get_config_path, set_config_status


@pytest.fixture(autouse=True)
def clean_configs(tmp_path, monkeypatch):
    """Use a temp directory for configs during tests."""
    test_configs = tmp_path / "configs"
    test_configs.mkdir()
    monkeypatch.setattr("automonai.backend.routers.configs.CONFIGS_DIR", test_configs)
    yield test_configs


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _save_config(client, name="test_config", model="unet", epochs="5"):
    command = (
        f"python3 -m automonai.core.run \\\n"
        f"  --dataset Dataset001 \\\n"
        f"  --model {model} \\\n"
        f"  --epochs {epochs}"
    )
    return client.post(
        "/api/configs/save",
        json={
            "name": name,
            "command": command,
            "params": {"model": model, "dataset": "Dataset001", "epochs": epochs},
        },
    )


class TestSaveConfig:
    def test_save_creates_file(self, client, clean_configs):
        resp = _save_config(client)
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert any(clean_configs.glob("*.json"))

    def test_save_creates_directory_if_missing(self, client, clean_configs):
        shutil.rmtree(clean_configs)
        assert not clean_configs.exists()
        resp = _save_config(client)
        assert resp.status_code == 200
        assert clean_configs.exists()

    def test_saved_config_has_correct_fields(self, client, clean_configs):
        _save_config(client, name="my_config", model="segresnet", epochs="10")
        path = get_config_path("my_config")
        data = json.loads(path.read_text())
        assert data["name"] == "my_config"
        assert "segresnet" in data["command"]
        assert data["params"]["model"] == "segresnet"
        assert len(data["launch_variants"]) == 7

    def test_saved_config_has_expected_variant_ids(self, client, clean_configs):
        _save_config(client, name="variants_cfg")
        data = json.loads(get_config_path("variants_cfg").read_text())
        variant_ids = [item["id"] for item in data["launch_variants"]]
        assert variant_ids == ["no_val", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "ensemble"]


class TestListConfigs:
    def test_list_empty(self, client):
        resp = client.get("/api/configs/list")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_saved(self, client):
        _save_config(client, name="cfg_a")
        _save_config(client, name="cfg_b")
        resp = client.get("/api/configs/list")
        names = [c["name"] for c in resp.json()]
        assert "cfg_a" in names
        assert "cfg_b" in names


class TestGetConfig:
    def test_get_existing(self, client):
        _save_config(client, name="my_cfg")
        resp = client.get("/api/configs/get/my_cfg")
        assert resp.status_code == 200
        assert resp.json()["name"] == "my_cfg"

    def test_get_missing_returns_404(self, client):
        resp = client.get("/api/configs/get/nonexistent")
        assert resp.status_code == 404

    def test_get_backfills_legacy_config_variants(self, client, clean_configs):
        legacy = {
            "name": "legacy_cfg",
            "command": "python3 -m automonai.core.run --dataset Dataset001 --model unet --epochs 5",
            "params": {"dataset": "Dataset001", "model": "unet", "epochs": "5"},
        }
        get_config_path("legacy_cfg").write_text(json.dumps(legacy, indent=2))

        resp = client.get("/api/configs/get/legacy_cfg")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["launch_variants"]) == 7
        assert data["launch_variants"][0]["id"] == "no_val"


class TestConfigStatus:
    def test_set_status(self, client, clean_configs):
        _save_config(client, name="status_test")
        set_config_status("status_test", "running")
        data = json.loads(get_config_path("status_test").read_text())
        assert data["status"] == "running"

    def test_set_status_done(self, client, clean_configs):
        _save_config(client, name="done_test")
        set_config_status("done_test", "done")
        data = json.loads(get_config_path("done_test").read_text())
        assert data["status"] == "done"

    def test_set_status_inferred(self, client, clean_configs):
        _save_config(client, name="infer_test")
        set_config_status("infer_test", "inferred")
        data = json.loads(get_config_path("infer_test").read_text())
        assert data["status"] == "inferred"


class TestDeleteConfig:
    def test_delete_existing(self, client, clean_configs):
        _save_config(client, name="del_me")
        resp = client.delete("/api/configs/delete/del_me")
        assert resp.status_code == 200
        assert not get_config_path("del_me").exists()

    def test_delete_missing_returns_404(self, client):
        resp = client.delete("/api/configs/delete/nonexistent")
        assert resp.status_code == 404
