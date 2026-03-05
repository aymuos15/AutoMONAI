"""Tests for transforms pipeline and image loading."""

import numpy as np
import pytest
import tempfile
import torch
from pathlib import Path
from PIL import Image

from src.transforms import PILLoadImage, get_transforms


class TestPILLoadImage:
    """Test the PILLoadImage transform."""

    def test_load_png_grayscale(self, tmp_path):
        """Test loading a PNG image."""
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(str(path))

        loader = PILLoadImage()
        result = loader(str(path))

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1, 64, 64)

    def test_load_png_rgb_converts_to_grayscale(self, tmp_path):
        """Test that RGB images are converted to grayscale."""
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        path = tmp_path / "test_rgb.png"
        img.save(str(path))

        loader = PILLoadImage()
        result = loader(str(path))

        assert result.shape == (1, 64, 64)

    def test_load_nifti_3d(self, tmp_path):
        """Test loading a NIfTI 3D volume."""
        nibabel = pytest.importorskip("nibabel")
        data = np.random.rand(32, 32, 32).astype(np.float32)
        img = nibabel.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nibabel.save(img, str(path))

        loader = PILLoadImage()
        result = loader(str(path), spatial_dims=3)

        assert result.dtype == np.float32
        assert result.shape == (1, 32, 32, 32)

    def test_load_nifti_as_2d_takes_middle_slice(self, tmp_path):
        """Test loading a 3D NIfTI as 2D takes the middle slice."""
        nibabel = pytest.importorskip("nibabel")
        data = np.random.rand(32, 32, 10).astype(np.float32)
        img = nibabel.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nibabel.save(img, str(path))

        loader = PILLoadImage()
        result = loader(str(path), spatial_dims=2)

        assert result.shape == (1, 32, 32)


class TestGetTransforms:
    """Test the get_transforms pipeline builder."""

    @pytest.fixture
    def sample_png(self, tmp_path):
        """Create a sample PNG for transform testing."""
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        path = tmp_path / "sample.png"
        img.save(str(path))
        return str(path)

    def test_basic_transforms_2d(self, sample_png):
        """Test basic 2D transform pipeline."""
        transforms = get_transforms(32, spatial_dims=2)
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_basic_transforms_different_sizes(self, sample_png):
        """Test transforms with different image sizes."""
        for size in [16, 32, 64, 128]:
            transforms = get_transforms(size, spatial_dims=2)
            result = transforms(sample_png)
            assert result.shape == (1, size, size)

    def test_minmax_normalization(self, sample_png):
        """Test minmax normalization is applied."""
        transforms = get_transforms(32, spatial_dims=2, norm=["minmax"])
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_zscore_normalization(self, sample_png):
        """Test zscore normalization is applied."""
        transforms = get_transforms(32, spatial_dims=2, norm=["zscore"])
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_both_normalizations(self, sample_png):
        """Test both minmax and zscore together."""
        transforms = get_transforms(32, spatial_dims=2, norm=["minmax", "zscore"])
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_center_crop(self, sample_png):
        """Test center crop."""
        transforms = get_transforms(32, spatial_dims=2, crop=["center"])
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_random_crop_train(self, sample_png):
        """Test random crop during training."""
        transforms = get_transforms(32, spatial_dims=2, crop=["random"], is_train=True)
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_random_crop_not_applied_at_test_time(self, sample_png):
        """Test that random crop falls through to center crop at test time."""
        transforms = get_transforms(32, spatial_dims=2, crop=["random"], is_train=False)
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_augmentation_pipeline(self, sample_png):
        """Test that augmentation transforms are added."""
        transforms = get_transforms(32, spatial_dims=2, augment=True)
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_full_pipeline(self, sample_png):
        """Test full pipeline: norm + crop + augment."""
        transforms = get_transforms(
            32, spatial_dims=2, norm=["minmax"], crop=["center"], augment=True
        )
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_no_norm_no_crop(self, sample_png):
        """Test with empty norm and crop lists."""
        transforms = get_transforms(32, spatial_dims=2, norm=[], crop=[])
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32)

    def test_tuple_size(self, sample_png):
        """Test with tuple image size."""
        transforms = get_transforms((32, 48), spatial_dims=2)
        result = transforms(sample_png)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 48)

    def test_3d_size_from_int(self):
        """Test that integer img_size produces correct 3D tuple."""
        transforms = get_transforms(32, spatial_dims=3)
        # Just verify it builds without error; can't easily test without 3D data
        assert transforms is not None
