"""Comprehensive tests for inferer factory and execution."""

import pytest
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer, PatchInferer, SaliencyInferer, SliceInferer

from automonai.core.inferers import get_inferer, run_inferer


class TestGetInferer:
    """Tests for the get_inferer factory function."""

    @pytest.mark.parametrize("inferer_name", ["simple", None])
    def test_simple_returns_none(self, inferer_name):
        """Test that 'simple' and None both return None."""
        result = get_inferer(inferer_name)
        assert result is None

    def test_sliding_window(self):
        """Test that 'sliding_window' returns a SlidingWindowInferer."""
        inferer = get_inferer("sliding_window")
        assert isinstance(inferer, SlidingWindowInferer)

    @pytest.mark.xfail(reason="PatchInferer string splitter not supported in this MONAI version")
    def test_patch(self):
        """Test that 'patch' returns a PatchInferer."""
        inferer = get_inferer("patch")
        assert isinstance(inferer, PatchInferer)

    def test_saliency(self):
        """Test that 'saliency' returns a SaliencyInferer."""
        inferer = get_inferer("saliency")
        assert isinstance(inferer, SaliencyInferer)

    def test_slice_2d(self):
        """Test that 'slice' returns a SliceInferer for 2D."""
        inferer = get_inferer("slice", spatial_dims=2)
        assert isinstance(inferer, SliceInferer)

    def test_slice_3d(self):
        """Test that 'slice' returns a SliceInferer for 3D."""
        inferer = get_inferer("slice", spatial_dims=3)
        assert isinstance(inferer, SliceInferer)

    def test_default_roi_size_2d(self):
        """Test default roi_size is (128, 128) for 2D sliding_window."""
        inferer = get_inferer("sliding_window", spatial_dims=2)
        assert inferer.roi_size == (128, 128)

    def test_default_roi_size_3d(self):
        """Test default roi_size is (128, 128, 128) for 3D sliding_window."""
        inferer = get_inferer("sliding_window", spatial_dims=3)
        assert inferer.roi_size == (128, 128, 128)

    def test_custom_roi_size(self):
        """Test that a custom roi_size is passed through."""
        inferer = get_inferer("sliding_window", roi_size=(64, 64))
        assert inferer.roi_size == (64, 64)

    def test_unknown_inferer_raises(self):
        """Test that an unknown inferer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown inferer"):
            get_inferer("nonexistent_inferer")

    @pytest.mark.parametrize(
        "inferer_name,expected_type",
        [
            ("sliding_window", SlidingWindowInferer),
            ("saliency", SaliencyInferer),
            ("slice", SliceInferer),
        ],
    )
    def test_all_inferer_types(self, inferer_name, expected_type):
        """Test that each valid inferer name returns the correct type."""
        inferer = get_inferer(inferer_name)
        assert isinstance(inferer, expected_type)

    @pytest.mark.xfail(reason="PatchInferer string splitter not supported in this MONAI version")
    def test_patch_inferer_type(self):
        """Test that 'patch' returns a PatchInferer."""
        inferer = get_inferer("patch")
        assert isinstance(inferer, PatchInferer)


class TestRunInferer:
    """Tests for the run_inferer execution function."""

    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dummy_model(self, device):
        """Create a simple convolutional model for testing."""
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
        ).to(device)
        model.eval()
        return model

    def test_none_inferer_calls_model_directly(self, device, dummy_model):
        """Test that None inferer returns model(inputs) directly."""
        inputs = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            expected = dummy_model(inputs)
            result = run_inferer(None, dummy_model, inputs)
        assert torch.equal(result, expected)

    def test_sliding_window_inferer_execution(self, device, dummy_model):
        """Test run_inferer with a real SlidingWindowInferer on dummy data."""
        inferer = get_inferer("sliding_window", roi_size=(32, 32))
        inputs = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            result = run_inferer(inferer, dummy_model, inputs)
        assert result.shape == (1, 2, 64, 64)
        assert not torch.isnan(result).any()

    def test_run_inferer_output_shape_matches_model(self, device, dummy_model):
        """Test that run_inferer with None produces same shape as direct call."""
        inputs = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            direct = dummy_model(inputs)
            via_inferer = run_inferer(None, dummy_model, inputs)
        assert direct.shape == via_inferer.shape
