"""Tests for RunLogger checkpoint and config management."""

import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from src.results import RunLogger


class TestRunLoggerCreation:
    """Test RunLogger directory and file creation."""

    def test_creates_run_directory(self, tmp_path):
        """Test that RunLogger creates the run directory."""
        logger = RunLogger("dataset1", "unet", results_root=str(tmp_path))

        assert logger.run_dir.exists()
        assert logger.checkpoints_dir.exists()

    def test_run_dir_structure(self, tmp_path):
        """Test directory structure: results_root/dataset/model/timestamp."""
        logger = RunLogger("dataset1", "unet", results_root=str(tmp_path))

        assert "dataset1" in str(logger.run_dir)
        assert "unet" in str(logger.run_dir)
        assert logger.checkpoints_dir == logger.run_dir / "checkpoints"

    def test_resume_reuses_directory(self, tmp_path):
        """Test that resume_from reuses the original run directory."""
        original = RunLogger("dataset1", "unet", results_root=str(tmp_path))
        original_dir = str(original.run_dir)

        resumed = RunLogger("dataset1", "unet", resume_from=original_dir)

        assert str(resumed.run_dir) == original_dir

    def test_stores_attributes(self, tmp_path):
        """Test that dataset_name, model_name, resume_from are stored."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))

        assert logger.dataset_name == "ds"
        assert logger.model_name == "model"
        assert logger.resume_from is None

    def test_stores_resume_from(self, tmp_path):
        """Test resume_from is stored when provided."""
        original = RunLogger("ds", "model", results_root=str(tmp_path))
        resumed = RunLogger("ds", "model", resume_from=str(original.run_dir))

        assert resumed.resume_from == str(original.run_dir)


class TestSaveConfig:
    """Test config saving and loading."""

    def test_save_config(self, tmp_path):
        """Test saving a config dict."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        config = {"dataset": "ds", "model": "model", "epochs": 10, "lr": 0.001}
        logger.save_config(config)

        config_path = logger.run_dir / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            loaded = json.load(f)
        assert loaded == config

    def test_save_config_overwrites(self, tmp_path):
        """Test that saving config overwrites the previous one."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_config({"epochs": 5})
        logger.save_config({"epochs": 10})

        with open(logger.run_dir / "config.json") as f:
            loaded = json.load(f)
        assert loaded["epochs"] == 10

    def test_load_run_config(self, tmp_path):
        """Test loading a config from a run directory."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        config = {"dataset": "ds", "model": "model"}
        logger.save_config(config)

        loaded = RunLogger.load_run_config(str(logger.run_dir))
        assert loaded == config

    def test_load_run_config_missing_raises(self, tmp_path):
        """Test that loading from a dir without config.json raises."""
        with pytest.raises(FileNotFoundError):
            RunLogger.load_run_config(str(tmp_path))


class TestCheckpoints:
    """Test checkpoint save and load."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_save_checkpoint(self, tmp_path, simple_model):
        """Test saving a checkpoint."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(simple_model, epoch=0)

        ckpt_path = logger.checkpoints_dir / "epoch_001.pt"
        assert ckpt_path.exists()

    def test_save_checkpoint_epoch_numbering(self, tmp_path, simple_model):
        """Test that epoch numbering is correct (epoch+1 in filename)."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(simple_model, epoch=0)
        logger.save_checkpoint(simple_model, epoch=4)
        logger.save_checkpoint(simple_model, epoch=9)

        assert (logger.checkpoints_dir / "epoch_001.pt").exists()
        assert (logger.checkpoints_dir / "epoch_005.pt").exists()
        assert (logger.checkpoints_dir / "epoch_010.pt").exists()

    def test_save_best_model(self, tmp_path, simple_model):
        """Test that is_best saves best_model.pt."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(simple_model, epoch=0, is_best=True)

        assert (logger.checkpoints_dir / "epoch_001.pt").exists()
        assert (logger.checkpoints_dir / "best_model.pt").exists()

    def test_save_with_optimizer(self, tmp_path, simple_model):
        """Test saving checkpoint with optimizer state."""
        optimizer = torch.optim.Adam(simple_model.parameters())
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(simple_model, epoch=0, optimizer=optimizer)

        ckpt = torch.load(logger.checkpoints_dir / "epoch_001.pt")
        assert "optimizer_state" in ckpt
        assert ckpt["optimizer_state"] is not None

    def test_save_without_optimizer(self, tmp_path, simple_model):
        """Test saving checkpoint without optimizer state."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(simple_model, epoch=0)

        ckpt = torch.load(logger.checkpoints_dir / "epoch_001.pt")
        assert "optimizer_state" not in ckpt or ckpt.get("optimizer_state") is None

    def test_load_checkpoint(self, tmp_path, simple_model):
        """Test loading a checkpoint."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(simple_model, epoch=2)

        loaded = RunLogger.load_checkpoint(str(logger.checkpoints_dir / "epoch_003.pt"))

        assert loaded["epoch"] == 3
        assert "model_state" in loaded
        assert isinstance(loaded["model_state"], dict)

    def test_load_checkpoint_restores_weights(self, tmp_path):
        """Test that loaded weights match saved weights."""
        model = nn.Linear(10, 5)
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(model, epoch=0)

        loaded = RunLogger.load_checkpoint(str(logger.checkpoints_dir / "epoch_001.pt"))
        model2 = nn.Linear(10, 5)
        model2.load_state_dict(loaded["model_state"])

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_get_checkpoint_path(self, tmp_path, simple_model):
        """Test get_checkpoint_path returns correct path."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))
        logger.save_checkpoint(simple_model, epoch=0, is_best=True)

        path = RunLogger.get_checkpoint_path(str(logger.run_dir), "best_model.pt")
        assert path.exists()
        assert path.name == "best_model.pt"

    def test_get_checkpoint_path_missing_raises(self, tmp_path):
        """Test that missing checkpoint raises FileNotFoundError."""
        logger = RunLogger("ds", "model", results_root=str(tmp_path))

        with pytest.raises(FileNotFoundError):
            RunLogger.get_checkpoint_path(str(logger.run_dir), "nonexistent.pt")


class TestFullWorkflow:
    """Test realistic save/resume workflows."""

    def test_train_save_resume_workflow(self, tmp_path):
        """Test a complete train -> save -> resume workflow."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Initial training run
        logger1 = RunLogger("ds", "unet", results_root=str(tmp_path))
        config = {"dataset": "ds", "model": "unet", "epochs": 10}
        logger1.save_config(config)
        logger1.save_checkpoint(model, epoch=4, optimizer=optimizer, is_best=True)

        # Resume
        logger2 = RunLogger("ds", "unet", resume_from=str(logger1.run_dir))
        loaded_config = RunLogger.load_run_config(str(logger2.run_dir))
        ckpt_path = RunLogger.get_checkpoint_path(str(logger2.run_dir), "best_model.pt")
        loaded_ckpt = RunLogger.load_checkpoint(str(ckpt_path))

        assert loaded_config["epochs"] == 10
        assert loaded_ckpt["epoch"] == 5

        # Continue training and save new checkpoint
        model.load_state_dict(loaded_ckpt["model_state"])
        logger2.save_checkpoint(model, epoch=7, is_best=True)

        assert (logger2.checkpoints_dir / "epoch_008.pt").exists()
        assert (logger2.checkpoints_dir / "best_model.pt").exists()
        # Original checkpoint still exists
        assert (logger2.checkpoints_dir / "epoch_005.pt").exists()
