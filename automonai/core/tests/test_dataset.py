"""Tests for dataset classes and factory functions."""

import numpy as np
import pytest
import torch
from monai.data import CacheDataset
from PIL import Image

from automonai.core.dataset import (
    DictTransform,
    TestDataset,
    TrainDataset,
    create_inference_dataset,
    create_train_dataset,
)


def _make_png(path):
    """Create a minimal 32x32 grayscale PNG file."""
    img = Image.fromarray(np.zeros((32, 32), dtype=np.uint8))
    img.save(path)


def _img_transform(x):
    return torch.tensor(np.array(Image.open(x).convert("L")), dtype=torch.float32).unsqueeze(0)


class TestDictTransform:
    """Tests for the DictTransform wrapper."""

    def test_applies_both_transforms(self):
        image_transform = lambda x: x * 2
        label_transform = lambda x: x + 10
        dt = DictTransform(image_transform, label_transform)

        result = dt({"image": 5, "label": 3})

        assert result["image"] == 10
        assert result["label"] == 13

    def test_preserves_keys(self):
        dt = DictTransform(lambda x: x, lambda x: x)

        result = dt({"image": "a", "label": "b"})

        assert set(result.keys()) == {"image", "label"}

    def test_with_tensor_transforms(self):
        image_transform = lambda x: torch.ones(1, 4, 4) * x
        label_transform = lambda x: torch.zeros(1, 4, 4) + x
        dt = DictTransform(image_transform, label_transform)

        result = dt({"image": 3.0, "label": 7.0})

        assert result["image"].shape == (1, 4, 4)
        assert torch.all(result["image"] == 3.0)
        assert result["label"].shape == (1, 4, 4)
        assert torch.all(result["label"] == 7.0)


class TestCreateTrainDataset:
    """Tests for the create_train_dataset factory function."""

    @pytest.fixture
    def train_files(self, tmp_path):
        """Create temp PNG files and return file dicts."""
        images_dir = tmp_path / "imagesTr"
        labels_dir = tmp_path / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        files = []
        for i in range(3):
            img_path = images_dir / f"img_{i:04d}_0000.png"
            lbl_path = labels_dir / f"img_{i:04d}.png"
            _make_png(img_path)
            _make_png(lbl_path)
            files.append({"image": str(img_path), "label": str(lbl_path)})
        return files

    @pytest.mark.parametrize("dataset_class_name", ["Dataset", "CacheDataset"])
    def test_returns_correct_type(self, dataset_class_name, train_files):
        expected_type = TrainDataset if dataset_class_name == "Dataset" else CacheDataset
        ds = create_train_dataset(
            dataset_class_name,
            train_files,
            transform=DictTransform(_img_transform, _img_transform),
            label_transform=_img_transform,
        )
        assert isinstance(ds, expected_type)

    def test_dataset_returns_train_dataset(self, train_files):
        ds = create_train_dataset(
            "Dataset",
            train_files,
            transform=_img_transform,
            label_transform=_img_transform,
        )
        assert isinstance(ds, TrainDataset)
        assert len(ds) == 3

        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert isinstance(sample["image"], torch.Tensor)

    def test_cache_dataset_len(self, train_files):
        ds = create_train_dataset(
            "CacheDataset",
            train_files,
            transform=DictTransform(_img_transform, _img_transform),
            label_transform=_img_transform,
            cache_rate=1.0,
        )
        assert len(ds) == 3

    def test_unknown_class_raises(self, train_files):
        with pytest.raises(ValueError, match="Unknown dataset class"):
            create_train_dataset(
                "NonExistentDataset",
                train_files,
                transform=_img_transform,
                label_transform=_img_transform,
            )


class TestCreateInferenceDataset:
    """Tests for the create_inference_dataset factory function."""

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create temp PNG files and return file dicts for inference."""
        images_dir = tmp_path / "imagesTs"
        images_dir.mkdir()

        files = []
        for i in range(3):
            img_path = images_dir / f"img_{i:04d}_0000.png"
            _make_png(img_path)
            files.append({"image": str(img_path)})
        return files

    def test_dataset_returns_test_dataset(self, test_files):
        ds = create_inference_dataset(
            "Dataset",
            test_files,
            transform=_img_transform,
        )
        assert isinstance(ds, TestDataset)
        assert len(ds) == 3

        sample = ds[0]
        assert "image" in sample
        assert "filename" in sample
        assert isinstance(sample["image"], torch.Tensor)

    def test_cache_dataset(self, test_files):
        transform = DictTransform(
            _img_transform,
            lambda x: x,
        )
        ds = create_inference_dataset(
            "CacheDataset",
            test_files,
            transform=transform,
        )
        assert isinstance(ds, CacheDataset)
        assert len(ds) == 3

    def test_unknown_class_raises(self, test_files):
        with pytest.raises(ValueError, match="Unknown dataset class"):
            create_inference_dataset(
                "NonExistentDataset",
                test_files,
                transform=_img_transform,
            )
