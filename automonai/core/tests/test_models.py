"""
Comprehensive tests for all medical image segmentation models.
Tests model instantiation, forward passes, and integration with the training pipeline.
"""

import pytest
import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.data import DataLoader

from automonai.core.models import get_model


class TestModelInstantiation:
    """Test that all models can be instantiated with various configurations."""

    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet", "swinunetr"])
    @pytest.mark.parametrize("spatial_dims", [2, 3])
    @pytest.mark.parametrize("in_channels", [1, 3])
    @pytest.mark.parametrize("out_channels", [2, 3, 4])
    def test_model_creation(self, model_name, spatial_dims, in_channels, out_channels):
        """Test that models can be instantiated with different configurations."""
        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            img_size=128,
        )
        assert model is not None
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet", "swinunetr"])
    def test_model_with_small_img_size(self, model_name):
        """Test models with different image sizes."""
        for img_size in [64, 128, 256]:
            model = get_model(
                model_name=model_name,
                in_channels=1,
                out_channels=2,
                spatial_dims=2,
                img_size=img_size,
            )
            assert model is not None


class TestForwardPass:
    """Test that all models produce correct output shapes."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet"])
    def test_2d_forward_pass(self, model_name, device):
        """Test 2D models with correct input shapes."""
        in_channels = 1
        out_channels = 3
        batch_size = 2
        img_size = 128

        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
            img_size=img_size,
        ).to(device)

        # Create dummy 2D input (B, C, H, W)
        x = torch.randn(batch_size, in_channels, img_size, img_size).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, out_channels, img_size, img_size)

    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet"])
    def test_3d_forward_pass(self, model_name, device):
        """Test 3D models with correct input shapes."""
        in_channels = 1
        out_channels = 3
        batch_size = 1
        img_size = 64

        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=3,
            img_size=img_size,
        ).to(device)

        # Create dummy 3D input (B, C, D, H, W)
        x = torch.randn(batch_size, in_channels, img_size, img_size, img_size).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, out_channels, img_size, img_size, img_size)

    def test_swinunetr_2d_forward_pass(self, device):
        """Test SwinUNETR with 2D input (requires larger input size)."""
        in_channels = 1
        out_channels = 3
        batch_size = 1
        img_size = 224  # SwinUNETR works well with 224x224

        model = get_model(
            model_name="swinunetr",
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
            img_size=img_size,
        ).to(device)

        x = torch.randn(batch_size, in_channels, img_size, img_size).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, out_channels, img_size, img_size)

    @pytest.mark.skip(reason="SwinUNETR 3D requires excessive GPU memory")
    def test_swinunetr_3d_forward_pass(self, device):
        """Test SwinUNETR with 3D input (skipped due to memory constraints)."""
        pass


class TestTrainingIntegration:
    """Test that all models work with the training pipeline."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dummy_dataset_2d(self):
        """Create a dummy 2D dataset."""
        num_samples = 8
        img_size = 128

        data = [
            {
                "image": torch.randn(1, img_size, img_size).numpy(),
                "label": torch.randint(0, 3, (1, img_size, img_size)).numpy(),
            }
            for _ in range(num_samples)
        ]
        return data

    @pytest.fixture
    def dummy_dataset_3d(self):
        """Create a dummy 3D dataset."""
        num_samples = 4
        img_size = 64

        data = [
            {
                "image": torch.randn(1, img_size, img_size, img_size).numpy(),
                "label": torch.randint(0, 3, (1, img_size, img_size, img_size)).numpy(),
            }
            for _ in range(num_samples)
        ]
        return data

    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet"])
    def test_2d_training_step(self, model_name, device, dummy_dataset_2d):
        """Test a complete training step for 2D models."""
        model = get_model(
            model_name=model_name,
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=128,
        ).to(device)

        loader = DataLoader(dummy_dataset_2d, batch_size=2, shuffle=False)
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()

            assert loss.item() > 0
            break  # Only test one batch

    @pytest.mark.parametrize("model_name", ["unet", "attention_unet", "segresnet"])
    def test_3d_training_step(self, model_name, device, dummy_dataset_3d):
        """Test a complete training step for 3D models."""
        model = get_model(
            model_name=model_name,
            in_channels=1,
            out_channels=3,
            spatial_dims=3,
            img_size=64,
        ).to(device)

        loader = DataLoader(dummy_dataset_3d, batch_size=1, shuffle=False)
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()

            assert loss.item() > 0
            break  # Only test one batch

    def test_swinunetr_2d_training_step(self, device, dummy_dataset_2d):
        """Test SwinUNETR training step with 2D data (requires larger inputs)."""
        model = get_model(
            model_name="swinunetr",
            in_channels=1,
            out_channels=3,
            spatial_dims=2,
            img_size=224,
        ).to(device)

        _loader = DataLoader(dummy_dataset_2d, batch_size=1, shuffle=False)
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        # Create proper batch with correct size
        x = torch.randn(1, 1, 224, 224).to(device)
        y = torch.randint(0, 3, (1, 1, 224, 224)).to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y.long())
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_swinunetr_3d_training_step(self, device, dummy_dataset_3d):
        """Test SwinUNETR training step with 3D data."""
        model = get_model(
            model_name="swinunetr",
            in_channels=1,
            out_channels=3,
            spatial_dims=3,
            img_size=64,
        ).to(device)

        loader = DataLoader(dummy_dataset_3d, batch_size=1, shuffle=False)
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()

            assert loss.item() > 0
            break  # Only test one batch


class TestModelComparison:
    """Test that all models produce reasonable outputs."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_all_models_inference_consistency(self, device):
        """Test that all models run inference without errors."""
        models_config = {
            "unet": 128,
            "attention_unet": 128,
            "segresnet": 128,
            "swinunetr": 224,  # SwinUNETR needs larger input
        }

        for model_name, img_size in models_config.items():
            batch_size = 1
            dummy_input = torch.randn(batch_size, 1, img_size, img_size).to(device)

            model = get_model(
                model_name=model_name,
                in_channels=1,
                out_channels=3,
                spatial_dims=2,
                img_size=img_size,
            ).to(device)

            model.eval()
            with torch.no_grad():
                output = model(dummy_input)

            assert output.shape == (batch_size, 3, img_size, img_size)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_models_have_learnable_parameters(self, device):
        """Ensure all models have learnable parameters."""
        model_names = ["unet", "attention_unet", "segresnet", "swinunetr"]

        for model_name in model_names:
            model = get_model(
                model_name=model_name,
                in_channels=1,
                out_channels=3,
                spatial_dims=2,
                img_size=128,
            ).to(device)

            params = list(model.parameters())
            assert len(params) > 0, f"{model_name} has no learnable parameters"

            total_params = sum(p.numel() for p in params if p.requires_grad)
            assert total_params > 0, f"{model_name} has no trainable parameters"

    def test_models_different_output_sizes(self, device):
        """Test models with different output channel sizes."""
        models_config = {
            "unet": 128,
            "attention_unet": 128,
            "segresnet": 128,
            "swinunetr": 224,  # SwinUNETR needs larger input
        }
        out_channels_list = [2, 5, 10]

        for out_channels in out_channels_list:
            for model_name, img_size in models_config.items():
                model = get_model(
                    model_name=model_name,
                    in_channels=1,
                    out_channels=out_channels,
                    spatial_dims=2,
                    img_size=img_size,
                ).to(device)

                dummy_input = torch.randn(1, 1, img_size, img_size).to(device)
                output = model(dummy_input)
                assert output.shape[1] == out_channels


class TestErrorHandling:
    """Test error handling for invalid configurations."""

    def test_invalid_model_name(self):
        """Test that invalid model names raise an error."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model(
                model_name="invalid_model",
                in_channels=1,
                out_channels=2,
                spatial_dims=2,
                img_size=128,
            )

    def test_zero_channels(self):
        """Test that zero channels raise an error or warning."""
        # Most models should fail with zero channels
        with pytest.raises((ValueError, RuntimeError)):
            model = get_model(
                model_name="unet",
                in_channels=0,
                out_channels=2,
                spatial_dims=2,
                img_size=128,
            )
            # Try to do a forward pass to trigger the error if not caught during init
            dummy_input = torch.randn(1, 0, 128, 128)
            model(dummy_input)
