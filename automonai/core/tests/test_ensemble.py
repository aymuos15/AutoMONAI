"""Tests for ensemble inference (ensemble_infer_with_metrics)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from automonai.core.inference import ensemble_infer_with_metrics
from automonai.core.train import get_metrics


class _TinyModel(nn.Module):
    """Minimal model that outputs a fixed pattern for deterministic testing."""

    def __init__(self, num_classes, bias_class=0):
        super().__init__()
        self.num_classes = num_classes
        self.bias_class = bias_class
        # Dummy parameter so it's a proper module
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B = x.shape[0]
        spatial = x.shape[2:]
        logits = torch.zeros(B, self.num_classes, *spatial, device=x.device)
        logits[:, self.bias_class] = 10.0 * self.w  # strongly predict bias_class
        return logits


@pytest.fixture
def fabric():
    from unittest.mock import MagicMock
    return MagicMock()


@pytest.fixture
def test_loader():
    """A simple labeled test loader with 4 samples, 2 classes.

    Labels are all class 1 (foreground) so DiceMetric(include_background=False)
    evaluates the foreground class meaningfully.
    """
    dataset = [
        {
            "image": torch.randn(1, 16, 16),
            "label": torch.ones(1, 16, 16, dtype=torch.long),  # all class 1 (foreground)
        }
        for _ in range(4)
    ]
    return torch.utils.data.DataLoader(dataset, batch_size=1)


class TestEnsembleMean:
    def test_returns_all_metrics(self, fabric, test_loader):
        models = [_TinyModel(2, bias_class=1) for _ in range(3)]
        results = ensemble_infer_with_metrics(
            fabric, models, test_loader,
            metric_names=["dice", "iou"],
            num_classes=2,
            method="mean",
        )
        assert "dice" in results
        assert "iou" in results
        assert isinstance(results["dice"], float)
        assert isinstance(results["iou"], float)

    def test_perfect_ensemble_high_dice(self, fabric, test_loader):
        """All models predict class 1, labels are class 1 -> high dice."""
        models = [_TinyModel(2, bias_class=1) for _ in range(3)]
        results = ensemble_infer_with_metrics(
            fabric, models, test_loader,
            metric_names=["dice"],
            num_classes=2,
            method="mean",
        )
        assert results["dice"] > 0.9

    def test_wrong_ensemble_low_dice(self, fabric):
        """All models predict class 0, labels are class 1 -> low dice."""
        dataset = [
            {
                "image": torch.randn(1, 16, 16),
                "label": torch.ones(1, 16, 16, dtype=torch.long),
            }
            for _ in range(4)
        ]
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        models = [_TinyModel(2, bias_class=0) for _ in range(3)]
        results = ensemble_infer_with_metrics(
            fabric, models, loader,
            metric_names=["dice"],
            num_classes=2,
            method="mean",
        )
        assert results["dice"] < 0.1

    def test_single_model_equals_non_ensemble(self, fabric, test_loader):
        """Ensemble with 1 model should match that model's predictions."""
        model = _TinyModel(2, bias_class=1)
        results = ensemble_infer_with_metrics(
            fabric, [model], test_loader,
            metric_names=["dice"],
            num_classes=2,
            method="mean",
        )
        assert results["dice"] > 0.9


class TestEnsembleVote:
    def test_majority_vote_returns_metrics(self, fabric, test_loader):
        models = [_TinyModel(2, bias_class=1) for _ in range(3)]
        results = ensemble_infer_with_metrics(
            fabric, models, test_loader,
            metric_names=["dice", "iou"],
            num_classes=2,
            method="vote",
        )
        assert "dice" in results
        assert "iou" in results

    def test_majority_correct(self, fabric, test_loader):
        """2 out of 3 models predict class 1 (correct) -> majority wins."""
        models = [
            _TinyModel(2, bias_class=1),
            _TinyModel(2, bias_class=1),
            _TinyModel(2, bias_class=0),
        ]
        results = ensemble_infer_with_metrics(
            fabric, models, test_loader,
            metric_names=["dice"],
            num_classes=2,
            method="vote",
        )
        assert results["dice"] > 0.9

    def test_majority_wrong(self, fabric, test_loader):
        """2 out of 3 models predict class 0 (wrong) -> dice drops."""
        models = [
            _TinyModel(2, bias_class=1),
            _TinyModel(2, bias_class=0),
            _TinyModel(2, bias_class=0),
        ]
        results = ensemble_infer_with_metrics(
            fabric, models, test_loader,
            metric_names=["dice"],
            num_classes=2,
            method="vote",
        )
        assert results["dice"] < 0.1


class TestEnsembleSavePredictions:
    def test_saves_png_files(self, fabric, test_loader, tmp_path):
        models = [_TinyModel(2, bias_class=1) for _ in range(2)]
        save_dir = str(tmp_path / "ensemble_preds")
        ensemble_infer_with_metrics(
            fabric, models, test_loader,
            metric_names=["dice"],
            num_classes=2,
            save_dir=save_dir,
            spatial_dims=2,
            method="mean",
        )
        from pathlib import Path
        pngs = list(Path(save_dir).glob("*.png"))
        assert len(pngs) == 4  # one per sample


class TestEnsembleWandBLog:
    def test_wandb_callback_called(self, fabric, test_loader):
        models = [_TinyModel(2, bias_class=1) for _ in range(2)]
        logged = []
        ensemble_infer_with_metrics(
            fabric, models, test_loader,
            metric_names=["dice"],
            num_classes=2,
            method="mean",
            wandb_log=lambda m: logged.append(m),
        )
        assert len(logged) == 4  # one per sample
        assert all("ensemble/dice" in entry for entry in logged)
