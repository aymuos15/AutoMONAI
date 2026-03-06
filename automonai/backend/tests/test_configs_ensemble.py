"""Tests for updated configs routes: val_split stripping, ensemble variant, fold_dirs injection."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from automonai.backend.routers.configs import (
    router,
    get_config_path,
    _strip_cv_flags,
    _build_launch_variants,
    set_fold_field,
    set_fold_status,
)


@pytest.fixture(autouse=True)
def clean_configs(tmp_path, monkeypatch):
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
    command = f"python3 -m automonai.core.run --dataset Dataset001 --model {model} --epochs {epochs}"
    return client.post(
        "/api/configs/save",
        json={
            "name": name,
            "command": command,
            "params": {"model": model, "dataset": "Dataset001", "epochs": epochs},
        },
    )


class TestStripCvFlags:
    def test_strips_cross_val(self):
        cmd = "python3 -m automonai.core.run --dataset D --cross_val 5 --cv_fold 2 --model unet"
        result = _strip_cv_flags(cmd)
        assert "--cross_val" not in result
        assert "--cv_fold" not in result
        assert "--dataset D" in result
        assert "--model unet" in result

    def test_strips_val_split(self):
        cmd = "python3 -m automonai.core.run --dataset D --val_split kfold --model unet"
        result = _strip_cv_flags(cmd)
        assert "--val_split" not in result

    def test_strips_ensemble_flags(self):
        cmd = "python3 -m automonai.core.run --ensemble_folds --ensemble_method mean --fold_dirs /a /b"
        result = _strip_cv_flags(cmd)
        assert "--ensemble_folds" not in result
        assert "--ensemble_method" not in result
        assert "--fold_dirs" not in result

    def test_strips_all_combined(self):
        cmd = (
            "python3 -m automonai.core.run --dataset D --cross_val 5 --cv_fold 3 "
            "--val_split kfold --ensemble_folds --ensemble_method vote"
        )
        result = _strip_cv_flags(cmd)
        assert "--cross_val" not in result
        assert "--cv_fold" not in result
        assert "--val_split" not in result
        assert "--ensemble_folds" not in result
        assert "--ensemble_method" not in result
        assert "--dataset D" in result


class TestBuildLaunchVariants:
    def test_includes_no_val_and_folds_and_ensemble(self):
        variants = _build_launch_variants("python3 -m automonai.core.run --dataset D --model unet", fold_count=3)
        ids = [v["id"] for v in variants]
        assert ids == ["no_val", "fold_1", "fold_2", "fold_3", "ensemble"]

    def test_fold_commands_include_val_split_kfold(self):
        variants = _build_launch_variants("python3 -m automonai.core.run --dataset D", fold_count=2)
        fold_variants = [v for v in variants if v["id"].startswith("fold_")]
        for v in fold_variants:
            assert "--val_split kfold" in v["command"]
            assert "--cross_val 2" in v["command"]

    def test_ensemble_command_includes_mode_infer(self):
        variants = _build_launch_variants("python3 -m automonai.core.run --dataset D", fold_count=3)
        ensemble = [v for v in variants if v["id"] == "ensemble"][0]
        assert "--mode infer" in ensemble["command"]
        assert "--ensemble_folds" in ensemble["command"]

    def test_no_val_command_is_clean(self):
        variants = _build_launch_variants("python3 -m automonai.core.run --dataset D", fold_count=3)
        no_val = variants[0]
        assert no_val["id"] == "no_val"
        assert "--cross_val" not in no_val["command"]
        assert "--val_split" not in no_val["command"]


class TestSavedConfigVariants:
    def test_saved_config_has_ensemble_variant(self, client, clean_configs):
        _save_config(client, name="ens_cfg")
        data = json.loads(get_config_path("ens_cfg").read_text())
        variant_ids = [v["id"] for v in data["launch_variants"]]
        assert "ensemble" in variant_ids

    def test_variant_count_is_folds_plus_no_val_plus_ensemble(self, client, clean_configs):
        resp = client.post(
            "/api/configs/save",
            json={
                "name": "count_cfg",
                "command": "python3 -m automonai.core.run --dataset D --model unet",
                "params": {},
                "cv": {"fold_count": 3},
            },
        )
        assert resp.status_code == 200
        data = json.loads(get_config_path("count_cfg").read_text())
        # 1 no_val + 3 folds + 1 ensemble = 5
        assert len(data["launch_variants"]) == 5


class TestFoldState:
    def test_set_and_read_fold_field(self, client, clean_configs):
        _save_config(client, name="fold_cfg")
        set_fold_field("fold_cfg", "fold_1", "run_dir", "/tmp/run1")
        data = json.loads(get_config_path("fold_cfg").read_text())
        assert data["fold_state"]["fold_1"]["run_dir"] == "/tmp/run1"

    def test_set_fold_status(self, client, clean_configs):
        _save_config(client, name="status_cfg")
        set_fold_status("status_cfg", "fold_2", "done")
        data = json.loads(get_config_path("status_cfg").read_text())
        assert data["fold_state"]["fold_2"]["status"] == "done"

    def test_fold_state_preserved_across_reads(self, client, clean_configs):
        _save_config(client, name="persist_cfg")
        set_fold_status("persist_cfg", "fold_1", "done")
        set_fold_field("persist_cfg", "fold_1", "run_dir", "/results/f1")
        set_fold_status("persist_cfg", "fold_2", "done")
        set_fold_field("persist_cfg", "fold_2", "run_dir", "/results/f2")

        # Read via API
        resp = client.get("/api/configs/get/persist_cfg")
        data = resp.json()
        assert data["fold_state"]["fold_1"]["status"] == "done"
        assert data["fold_state"]["fold_1"]["run_dir"] == "/results/f1"
        assert data["fold_state"]["fold_2"]["status"] == "done"
