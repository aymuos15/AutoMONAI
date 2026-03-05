"""Tests for training utilities: metrics, loss factory, and train_one_epoch."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric, MeanIoU

from src.train import get_loss, get_metrics, compute_metrics, get_metric_values, train_one_epoch


class TestGetMetrics:
    """Test metric factory function."""

    def test_dice_metric(self):
        """Test creating dice metric."""
        metrics = get_metrics(["dice"])

        assert "dice" in metrics
        assert isinstance(metrics["dice"], DiceMetric)

    def test_iou_metric(self):
        """Test creating IoU metric."""
        metrics = get_metrics(["iou"])

        assert "iou" in metrics
        assert isinstance(metrics["iou"], MeanIoU)

    def test_both_metrics(self):
        """Test creating both metrics."""
        metrics = get_metrics(["dice", "iou"])

        assert len(metrics) == 2
        assert "dice" in metrics
        assert "iou" in metrics

    def test_empty_list(self):
        """Test with empty metric list."""
        metrics = get_metrics([])
        assert metrics == {}

    def test_unknown_metric_ignored(self):
        """Test that unknown metrics are silently ignored."""
        metrics = get_metrics(["unknown"])
        assert metrics == {}


class TestComputeAndGetMetrics:
    """Test compute_metrics and get_metric_values together."""

    def _make_onehot(self, labels, num_classes):
        """Helper to create one-hot encoded tensors."""
        onehot = F.one_hot(labels.squeeze(1).long(), num_classes)
        return onehot.permute(0, 3, 1, 2).float()

    def test_dice_metric_computation(self):
        """Test dice metric produces a valid scalar."""
        metrics = get_metrics(["dice"])
        num_classes = 3

        # Create predictions (softmax-like) and labels
        preds = torch.rand(2, num_classes, 16, 16)
        preds = (preds == preds.max(dim=1, keepdim=True).values).float()

        labels = torch.randint(0, num_classes, (2, 1, 16, 16))
        labels_onehot = self._make_onehot(labels, num_classes)

        compute_metrics(metrics, preds, labels_onehot)
        results = get_metric_values(metrics)

        assert "dice" in results
        assert isinstance(results["dice"], float)
        assert 0.0 <= results["dice"] <= 1.0

    def test_iou_metric_computation(self):
        """Test IoU metric produces a valid scalar."""
        metrics = get_metrics(["iou"])
        num_classes = 3

        preds = torch.rand(2, num_classes, 16, 16)
        preds = (preds == preds.max(dim=1, keepdim=True).values).float()

        labels = torch.randint(0, num_classes, (2, 1, 16, 16))
        labels_onehot = self._make_onehot(labels, num_classes)

        compute_metrics(metrics, preds, labels_onehot)
        results = get_metric_values(metrics)

        assert "iou" in results
        assert isinstance(results["iou"], float)
        assert 0.0 <= results["iou"] <= 1.0

    def test_perfect_prediction_high_dice(self):
        """Test that perfect predictions produce high dice score."""
        metrics = get_metrics(["dice"])
        num_classes = 3

        # Create labels and make predictions match exactly
        labels = torch.randint(0, num_classes, (2, 1, 16, 16))
        labels_onehot = self._make_onehot(labels, num_classes)

        compute_metrics(metrics, labels_onehot, labels_onehot)
        results = get_metric_values(metrics)

        assert results["dice"] > 0.9

    def test_metrics_reset_after_get(self):
        """Test that get_metric_values resets the metrics."""
        metrics = get_metrics(["dice"])
        num_classes = 3

        preds = torch.rand(2, num_classes, 16, 16)
        preds = (preds == preds.max(dim=1, keepdim=True).values).float()
        labels = torch.randint(0, num_classes, (2, 1, 16, 16))
        labels_onehot = self._make_onehot(labels, num_classes)

        compute_metrics(metrics, preds, labels_onehot)
        results1 = get_metric_values(metrics)

        # After reset, computing with new data should give independent results
        compute_metrics(metrics, labels_onehot, labels_onehot)
        results2 = get_metric_values(metrics)

        # Perfect prediction after reset should give high dice
        assert results2["dice"] > 0.9

    def test_multiple_batches_accumulated(self):
        """Test that metrics accumulate across multiple compute calls."""
        metrics = get_metrics(["dice"])
        num_classes = 3

        for _ in range(3):
            preds = torch.rand(2, num_classes, 16, 16)
            preds = (preds == preds.max(dim=1, keepdim=True).values).float()
            labels = torch.randint(0, num_classes, (2, 1, 16, 16))
            labels_onehot = self._make_onehot(labels, num_classes)
            compute_metrics(metrics, preds, labels_onehot)

        results = get_metric_values(metrics)
        assert "dice" in results
        assert isinstance(results["dice"], float)


class TestTrainOneEpoch:
    """Test train_one_epoch function."""

    @pytest.fixture
    def fabric(self):
        """Create a mock Fabric instance."""
        mock = MagicMock()
        mock.backward = lambda loss: loss.backward()
        return mock

    @pytest.fixture
    def setup(self):
        """Create model, loader, loss, optimizer for training."""
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

        # Create a simple dataloader
        dataset = [
            {
                "image": torch.randn(1, 32, 32),
                "label": torch.randint(0, 3, (1, 32, 32)),
            }
            for _ in range(4)
        ]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        return model, loader, loss_fn, optimizer

    def test_returns_loss(self, fabric, setup):
        """Test that train_one_epoch returns a loss value."""
        model, loader, loss_fn, optimizer = setup
        result = train_one_epoch(fabric, model, loader, loss_fn, optimizer)

        assert "loss" in result
        assert isinstance(result["loss"], float)
        assert result["loss"] > 0

    def test_returns_dice_metric(self, fabric, setup):
        """Test that dice metric is returned when requested."""
        model, loader, loss_fn, optimizer = setup
        result = train_one_epoch(fabric, model, loader, loss_fn, optimizer, metrics=["dice"])

        assert "loss" in result
        assert "dice" in result
        assert isinstance(result["dice"], float)

    def test_returns_iou_metric(self, fabric, setup):
        """Test that IoU metric is returned when requested."""
        model, loader, loss_fn, optimizer = setup
        result = train_one_epoch(fabric, model, loader, loss_fn, optimizer, metrics=["iou"])

        assert "iou" in result

    def test_returns_both_metrics(self, fabric, setup):
        """Test that both metrics are returned when requested."""
        model, loader, loss_fn, optimizer = setup
        result = train_one_epoch(
            fabric, model, loader, loss_fn, optimizer, metrics=["dice", "iou"]
        )

        assert "loss" in result
        assert "dice" in result
        assert "iou" in result

    def test_no_metrics(self, fabric, setup):
        """Test training without metrics."""
        model, loader, loss_fn, optimizer = setup
        result = train_one_epoch(fabric, model, loader, loss_fn, optimizer, metrics=None)

        assert "loss" in result
        assert "dice" not in result
        assert "iou" not in result

    def test_cross_entropy_loss(self, fabric):
        """Test training with CrossEntropyLoss."""
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        dataset = [
            {
                "image": torch.randn(1, 32, 32),
                "label": torch.randint(0, 3, (1, 32, 32)),
            }
            for _ in range(4)
        ]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        result = train_one_epoch(fabric, model, loader, loss_fn, optimizer)
        assert result["loss"] > 0

    def test_model_weights_change(self, fabric, setup):
        """Test that model weights actually change after training."""
        model, loader, loss_fn, optimizer = setup

        initial_params = [p.clone() for p in model.parameters()]
        train_one_epoch(fabric, model, loader, loss_fn, optimizer)
        final_params = list(model.parameters())

        changed = any(
            not torch.equal(init, final)
            for init, final in zip(initial_params, final_params)
        )
        assert changed, "Model weights should change after training"

    def test_focal_loss(self, fabric):
        """Test training with FocalLoss."""
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = FocalLoss(to_onehot_y=True, use_softmax=True)

        dataset = [
            {
                "image": torch.randn(1, 32, 32),
                "label": torch.randint(0, 3, (1, 32, 32)),
            }
            for _ in range(4)
        ]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        result = train_one_epoch(fabric, model, loader, loss_fn, optimizer)
        assert result["loss"] > 0
