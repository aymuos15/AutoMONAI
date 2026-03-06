"""
Integration tests for the entire MonaiUI pipeline.
Tests CLI, config, dataset handling, and end-to-end workflows.
"""

import pytest
import torch
from pathlib import Path

from automonai.core.config import MODELS, DATASETS
from automonai.core.cli import get_parser
from automonai.core.dataset import list_datasets
from automonai.core.models import get_model


class TestConfig:
    """Test configuration and available options."""

    def test_models_in_config(self):
        """Test that all expected models are in config."""
        expected_models = ["unet", "attention_unet", "segresnet", "swinunetr"]
        for model_name in expected_models:
            assert model_name in MODELS
            assert "name" in MODELS[model_name]
            assert "description" in MODELS[model_name]

    def test_all_models_have_descriptions(self):
        """Test that all models have meaningful descriptions."""
        for model_name, model_info in MODELS.items():
            assert model_info["name"]
            assert model_info["description"]
            assert len(model_info["description"]) > 5

    def test_datasets_loaded(self):
        """Test that datasets can be loaded."""
        datasets = DATASETS
        # Should have at least some datasets
        assert isinstance(datasets, dict)

    def test_get_model_for_each_config(self):
        """Test that get_model works for each model in config."""
        for model_name in MODELS.keys():
            model = get_model(model_name, 1, 2, 2, 128)
            assert model is not None


class TestCLI:
    """Test command-line interface."""

    def test_parser_creation(self):
        """Test that argument parser can be created."""
        parser = get_parser()
        assert parser is not None

    def test_parser_default_values(self):
        """Test that parser has sensible defaults."""
        parser = get_parser()
        args = parser.parse_args([])
        assert args.model == "unet"
        assert args.epochs > 0
        assert args.batch_size > 0
        assert args.lr > 0

    def test_parser_model_choices(self):
        """Test that all models are available in CLI."""
        parser = get_parser()
        # Get the model argument
        model_action = None
        for action in parser._actions:
            if action.dest == "model":
                model_action = action
                break

        assert model_action is not None
        expected_choices = list(MODELS.keys())
        assert model_action.choices == expected_choices

    def test_parser_accepts_valid_models(self):
        """Test that parser accepts all valid models."""
        parser = get_parser()
        for model_name in MODELS.keys():
            args = parser.parse_args(["--model", model_name])
            assert args.model == model_name

    def test_parser_rejects_invalid_model(self):
        """Test that parser rejects invalid models."""
        parser = get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "invalid_model"])

    def test_parser_spatial_dims_choices(self):
        """Test that parser accepts valid spatial dimensions."""
        parser = get_parser()
        args_2d = parser.parse_args(["--spatial_dims", "2"])
        args_3d = parser.parse_args(["--spatial_dims", "3"])
        assert args_2d.spatial_dims == 2
        assert args_3d.spatial_dims == 3

    def test_parser_rejects_invalid_spatial_dims(self):
        """Test that parser rejects invalid spatial dimensions."""
        parser = get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--spatial_dims", "4"])

    def test_parser_device_choices(self):
        """Test that parser accepts valid device options."""
        parser = get_parser()
        args_cuda = parser.parse_args(["--device", "cuda"])
        args_cpu = parser.parse_args(["--device", "cpu"])
        assert args_cuda.device == "cuda"
        assert args_cpu.device == "cpu"

    def test_parser_numeric_arguments(self):
        """Test that numeric arguments are parsed correctly."""
        parser = get_parser()
        args = parser.parse_args(
            [
                "--epochs",
                "10",
                "--batch_size",
                "32",
                "--lr",
                "0.001",
                "--img_size",
                "256",
                "--num_workers",
                "4",
            ]
        )
        assert args.epochs == 10
        assert args.batch_size == 32
        assert args.lr == 0.001
        assert args.img_size == 256
        assert args.num_workers == 4


class TestDataset:
    """Test dataset functionality."""

    def test_list_datasets_returns_list(self):
        """Test that list_datasets returns a list."""
        datasets = list_datasets()
        assert isinstance(datasets, list)

    def test_datasets_have_required_fields(self):
        """Test that all datasets have required fields."""
        for dataset_name, dataset_info in DATASETS.items():
            assert "name" in dataset_info
            assert "description" in dataset_info
            assert "channels" in dataset_info
            assert "labels" in dataset_info


class TestModelConfiguration:
    """Test that models work with all dataset configurations."""

    def test_model_with_various_channel_counts(self):
        """Test models with different input/output channel counts."""
        models_config = {
            "unet": 128,
            "attention_unet": 128,
            "segresnet": 128,
            "swinunetr": 224,  # SwinUNETR needs larger input
        }
        for model_name, img_size in models_config.items():
            # Test with different input channels (typical for medical imaging)
            for in_channels in [1, 3]:
                # Test with different output channels (different label counts)
                for out_channels in [2, 3, 5]:
                    model = get_model(
                        model_name=model_name,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        spatial_dims=2,
                        img_size=img_size,
                    )
                    assert model is not None

                    # Verify forward pass
                    x = torch.randn(1, in_channels, img_size, img_size)
                    with torch.no_grad():
                        y = model(x)
                    assert y.shape == (1, out_channels, img_size, img_size)

    def test_model_with_datasets(self):
        """Test that models can be created for each dataset."""
        for model_name in MODELS.keys():
            for dataset_name, dataset_info in DATASETS.items():
                in_channels = len(dataset_info.get("channels", ["MRI"]))
                out_channels = len(dataset_info.get("labels", {"background": 0}))

                model = get_model(
                    model_name=model_name,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    spatial_dims=2,
                    img_size=128,
                )
                assert model is not None


class TestEndToEnd:
    """Test end-to-end workflows."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_model_creation_and_inference(self, device):
        """Test creating a model and running inference."""
        # Simulate what run.py does
        dataset_name = "Dataset001_Cellpose"
        model_name = "unet"

        # Get dataset info
        if dataset_name in DATASETS:
            dataset_info = DATASETS[dataset_name]
            in_channels = len(dataset_info.get("channels", ["MRI"]))
            out_channels = len(dataset_info.get("labels", {"background": 0}))
        else:
            in_channels = 1
            out_channels = 2

        # Create model
        model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
            img_size=128,
        ).to(device)

        # Run inference
        dummy_input = torch.randn(2, in_channels, 128, 128).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, out_channels, 128, 128)

    def test_all_models_can_be_trained(self, device):
        """Test that all models can be used in a training loop."""
        from monai.losses import DiceLoss

        models_config = {
            "unet": 128,
            "attention_unet": 128,
            "segresnet": 128,
            "swinunetr": 224,  # SwinUNETR needs larger input
        }
        for model_name, img_size in models_config.items():
            model = get_model(
                model_name=model_name,
                in_channels=1,
                out_channels=3,
                spatial_dims=2,
                img_size=img_size,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

            # Create dummy batch
            x = torch.randn(1, 1, img_size, img_size).to(device)
            y = torch.randint(0, 3, (1, 1, img_size, img_size)).to(device)

            # Training step
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y.long())
            loss.backward()
            optimizer.step()

            assert loss.item() > 0
            assert not torch.isnan(loss)

    def test_model_save_and_load(self, device):
        """Test that models can be saved and loaded."""
        import tempfile

        model_name = "unet"
        model = get_model(
            model_name=model_name,
            in_channels=1,
            out_channels=2,
            spatial_dims=2,
            img_size=128,
        ).to(device)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            checkpoint_path = f.name

        # Load and verify
        model_loaded = get_model(
            model_name=model_name,
            in_channels=1,
            out_channels=2,
            spatial_dims=2,
            img_size=128,
        ).to(device)
        model_loaded.load_state_dict(torch.load(checkpoint_path))

        # Verify they produce the same output
        x = torch.randn(1, 1, 128, 128).to(device)
        model.eval()
        model_loaded.eval()
        with torch.no_grad():
            y1 = model(x)
            y2 = model_loaded(x)

        assert torch.allclose(y1, y2, rtol=1e-5, atol=1e-6)

        # Cleanup
        Path(checkpoint_path).unlink()


class TestModelWeights:
    """Test that models have appropriate weight initialization."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_models_have_different_initial_weights(self, device):
        """Test that different model instances have different weights."""
        model1 = get_model("unet", 1, 2, 2, 128).to(device)
        model2 = get_model("unet", 1, 2, 2, 128).to(device)

        params1 = list(model1.parameters())[0]
        params2 = list(model2.parameters())[0]

        # Weights should not be identical (extremely unlikely if randomly initialized)
        assert not torch.allclose(params1, params2)

    def test_model_gradients_enabled_by_default(self, device):
        """Test that model parameters have gradients enabled."""
        model = get_model("unet", 1, 2, 2, 128).to(device)

        for param in model.parameters():
            assert param.requires_grad
