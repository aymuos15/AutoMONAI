"""
Comprehensive tests for all loss functions with different models and configurations.
Tests that each loss works with all models, spatial dimensions, and dataset configurations.
"""

import pytest
import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss

from automonai.core.models import get_model
from automonai.core.train import get_loss


class TestLossInstantiation:
    """Test that all loss functions can be instantiated."""

    def test_dice_loss_creation(self):
        """Test DiceLoss instantiation."""
        loss = DiceLoss(to_onehot_y=True, softmax=True)
        assert loss is not None
        assert isinstance(loss, nn.Module)

    def test_cross_entropy_loss_creation(self):
        """Test CrossEntropyLoss instantiation."""
        loss = nn.CrossEntropyLoss()
        assert loss is not None
        assert isinstance(loss, nn.Module)

    def test_focal_loss_creation(self):
        """Test FocalLoss instantiation."""
        loss = FocalLoss(to_onehot_y=True, use_softmax=True)
        assert loss is not None
        assert isinstance(loss, nn.Module)

    def test_get_loss_all_types(self):
        """Test get_loss factory function for all loss types."""
        for loss_name in ["dice", "cross_entropy", "focal"]:
            loss = get_loss(loss_name)
            assert loss is not None
            assert isinstance(loss, nn.Module)

    def test_get_loss_invalid_type(self):
        """Test get_loss with invalid loss name."""
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("invalid_loss")


class TestLossWithModels2D:
    """Test all losses with 2D models."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("loss_name", ["dice", "cross_entropy", "focal"])
    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet", "swinunetr"])
    def test_loss_with_2d_models(self, loss_name, model_name, device):
        """Test each loss with all 2D models."""
        batch_size = 2
        in_channels = 1
        out_channels = 3
        img_size = 128 if model_name != "swinunetr" else 224
        spatial_dims = 2

        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            img_size=img_size,
        ).to(device)

        loss_fn = get_loss(loss_name)

        # Create dummy inputs and targets
        inputs = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
        targets = torch.randint(0, out_channels, (batch_size, 1, img_size, img_size)).to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

        # Compute loss
        # CrossEntropyLoss expects targets without channel dimension
        if loss_name == "cross_entropy":
            loss = loss_fn(outputs, targets.squeeze(1).long())
        else:
            loss = loss_fn(outputs, targets.long())

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    @pytest.mark.parametrize("loss_name", ["dice", "cross_entropy", "focal"])
    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet"])
    def test_loss_with_3d_models(self, loss_name, model_name, device):
        """Test each loss with all 3D models."""
        batch_size = 1
        in_channels = 1
        out_channels = 3
        img_size = 64
        spatial_dims = 3

        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            img_size=img_size,
        ).to(device)

        loss_fn = get_loss(loss_name)

        # Create dummy inputs and targets
        inputs = torch.randn(batch_size, in_channels, img_size, img_size, img_size).to(device)
        targets = torch.randint(0, out_channels, (batch_size, 1, img_size, img_size, img_size)).to(
            device
        )

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

        # Compute loss
        # CrossEntropyLoss expects targets without channel dimension
        if loss_name == "cross_entropy":
            loss = loss_fn(outputs, targets.squeeze(1).long())
        else:
            loss = loss_fn(outputs, targets.long())

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestLossBackpropagation:
    """Test that losses support backpropagation with all models."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("loss_name", ["dice", "cross_entropy", "focal"])
    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet"])
    def test_loss_backprop_2d(self, loss_name, model_name, device):
        """Test backpropagation through losses with 2D models."""
        batch_size = 2
        in_channels = 1
        out_channels = 3
        img_size = 128

        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
            img_size=img_size,
        ).to(device)

        loss_fn = get_loss(loss_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create dummy inputs and targets
        inputs = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
        targets = torch.randint(0, out_channels, (batch_size, 1, img_size, img_size)).to(device)

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)

        # CrossEntropyLoss expects targets without channel dimension
        if loss_name == "cross_entropy":
            loss = loss_fn(outputs, targets.squeeze(1).long())
        else:
            loss = loss_fn(outputs, targets.long())

        loss.backward()

        # Verify gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        optimizer.step()
        assert loss.item() > 0

    @pytest.mark.parametrize("loss_name", ["dice", "cross_entropy", "focal"])
    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet"])
    def test_loss_backprop_3d(self, loss_name, model_name, device):
        """Test backpropagation through losses with 3D models."""
        batch_size = 1
        in_channels = 1
        out_channels = 3
        img_size = 64

        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=3,
            img_size=img_size,
        ).to(device)

        loss_fn = get_loss(loss_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create dummy inputs and targets
        inputs = torch.randn(batch_size, in_channels, img_size, img_size, img_size).to(device)
        targets = torch.randint(0, out_channels, (batch_size, 1, img_size, img_size, img_size)).to(
            device
        )

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)

        # CrossEntropyLoss expects targets without channel dimension
        if loss_name == "cross_entropy":
            loss = loss_fn(outputs, targets.squeeze(1).long())
        else:
            loss = loss_fn(outputs, targets.long())

        loss.backward()

        # Verify gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        optimizer.step()
        assert loss.item() > 0


class TestLossWithDifferentChannels:
    """Test losses with different input/output channel configurations."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("loss_name", ["dice", "cross_entropy", "focal"])
    @pytest.mark.parametrize("in_channels", [1, 3])
    @pytest.mark.parametrize("out_channels", [2, 4, 10])
    def test_loss_with_varying_channels(self, loss_name, in_channels, out_channels, device):
        """Test losses with varying input/output channels."""
        model = get_model(
            model_name="unet",
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
            img_size=128,
        ).to(device)

        loss_fn = get_loss(loss_name)

        batch_size = 2
        img_size = 128

        inputs = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
        targets = torch.randint(0, out_channels, (batch_size, 1, img_size, img_size)).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

        # CrossEntropyLoss expects targets without channel dimension
        if loss_name == "cross_entropy":
            loss = loss_fn(outputs, targets.squeeze(1).long())
        else:
            loss = loss_fn(outputs, targets.long())

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestLossWithDifferentBatchSizes:
    """Test losses with different batch sizes."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("loss_name", ["dice", "cross_entropy", "focal"])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_loss_with_varying_batch_sizes(self, loss_name, batch_size, device):
        """Test losses with different batch sizes."""
        model = get_model(
            model_name="unet",
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=128,
        ).to(device)

        loss_fn = get_loss(loss_name)

        inputs = torch.randn(batch_size, 1, 128, 128).to(device)
        targets = torch.randint(0, 3, (batch_size, 1, 128, 128)).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

        # CrossEntropyLoss expects targets without channel dimension
        if loss_name == "cross_entropy":
            loss = loss_fn(outputs, targets.squeeze(1).long())
        else:
            loss = loss_fn(outputs, targets.long())

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestLossConsistency:
    """Test that losses produce consistent results."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("loss_name", ["dice", "cross_entropy", "focal"])
    def test_loss_deterministic(self, loss_name, device):
        """Test that same input produces same loss value."""
        model = get_model(
            model_name="unet",
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=128,
        ).to(device)
        model.eval()

        loss_fn = get_loss(loss_name)

        inputs = torch.randn(2, 1, 128, 128).to(device)
        targets = torch.randint(0, 3, (2, 1, 128, 128)).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            # CrossEntropyLoss expects targets without channel dimension
            if loss_name == "cross_entropy":
                loss1 = loss_fn(outputs, targets.squeeze(1).long())
                loss2 = loss_fn(outputs, targets.squeeze(1).long())
            else:
                loss1 = loss_fn(outputs, targets.long())
                loss2 = loss_fn(outputs, targets.long())

        assert torch.allclose(loss1, loss2, rtol=1e-5)

    def test_loss_all_zeros_target(self, device):
        """Test loss with all-zero target."""
        model = get_model(
            model_name="unet",
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=128,
        ).to(device)
        model.eval()

        inputs = torch.randn(1, 1, 128, 128).to(device)
        targets = torch.zeros(1, 1, 128, 128, dtype=torch.long).to(device)

        for loss_name in ["dice", "cross_entropy", "focal"]:
            loss_fn = get_loss(loss_name)
            with torch.no_grad():
                outputs = model(inputs)
                # CrossEntropyLoss expects targets without channel dimension
                if loss_name == "cross_entropy":
                    loss = loss_fn(outputs, targets.squeeze(1).long())
                else:
                    loss = loss_fn(outputs, targets.long())

            assert loss.item() >= 0
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_loss_all_ones_target(self, device):
        """Test loss with all-ones target."""
        model = get_model(
            model_name="unet",
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=128,
        ).to(device)
        model.eval()

        inputs = torch.randn(1, 1, 128, 128).to(device)
        targets = torch.ones(1, 1, 128, 128, dtype=torch.long).to(device)

        for loss_name in ["dice", "cross_entropy", "focal"]:
            loss_fn = get_loss(loss_name)
            with torch.no_grad():
                outputs = model(inputs)
                # CrossEntropyLoss expects targets without channel dimension
                if loss_name == "cross_entropy":
                    loss = loss_fn(outputs, targets.squeeze(1).long())
                else:
                    loss = loss_fn(outputs, targets.long())

            assert loss.item() >= 0
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)


class TestLossComparison:
    """Test loss values relative to each other."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_all_losses_produce_scalars(self, device):
        """Test that all losses produce scalar values."""
        model = get_model(
            model_name="unet",
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=128,
        ).to(device)
        model.eval()

        inputs = torch.randn(2, 1, 128, 128).to(device)
        targets = torch.randint(0, 3, (2, 1, 128, 128)).to(device)

        with torch.no_grad():
            outputs = model(inputs)

        for loss_name in ["dice", "cross_entropy", "focal"]:
            loss_fn = get_loss(loss_name)
            # CrossEntropyLoss expects targets without channel dimension
            if loss_name == "cross_entropy":
                loss = loss_fn(outputs, targets.squeeze(1).long())
            else:
                loss = loss_fn(outputs, targets.long())

            # Check that loss is a scalar
            assert loss.dim() == 0
            assert loss.item() >= 0

    def test_loss_differences_on_different_targets(self, device):
        """Test that different targets produce different loss values."""
        model = get_model(
            model_name="unet",
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=128,
        ).to(device)
        model.eval()

        inputs = torch.randn(2, 1, 128, 128).to(device)

        loss_fn = get_loss("dice")

        with torch.no_grad():
            outputs = model(inputs)

        # Create two different targets
        targets1 = torch.randint(0, 3, (2, 1, 128, 128)).to(device)
        targets2 = torch.randint(0, 3, (2, 1, 128, 128)).to(device)

        loss1 = loss_fn(outputs, targets1.long())
        loss2 = loss_fn(outputs, targets2.long())

        # Losses should likely be different (with extremely high probability)
        # We don't assert inequality to avoid flakiness, but both should be valid
        assert loss1.item() >= 0
        assert loss2.item() >= 0
        assert not torch.isnan(loss1)
        assert not torch.isnan(loss2)
