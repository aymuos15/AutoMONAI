"""Tests for data splitting (split_train_val) and validation loop (validate)."""

import json
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock
from monai.losses import DiceLoss

from automonai.core.dataset import split_train_val
from automonai.core.train import validate


# ── split_train_val ──────────────────────────────────────────────────────────


def _make_files(n):
    """Create n dummy file dicts."""
    return [{"image": f"/data/img_{i:04d}.png", "label": f"/data/lbl_{i:04d}.png"} for i in range(n)]


class TestSplitNone:
    def test_returns_all_train_no_val(self):
        files = _make_files(10)
        train, val = split_train_val(files, mode="none")
        assert train == files
        assert val == []


class TestSplitHoldout:
    def test_split_sizes(self):
        files = _make_files(20)
        train, val = split_train_val(files, mode="holdout", val_ratio=0.2, seed=42)
        assert len(train) + len(val) == 20
        assert len(val) == 4  # 20 * 0.2

    def test_no_overlap(self):
        files = _make_files(30)
        train, val = split_train_val(files, mode="holdout", val_ratio=0.3, seed=0)
        train_imgs = {f["image"] for f in train}
        val_imgs = {f["image"] for f in val}
        assert train_imgs & val_imgs == set()

    def test_covers_all_files(self):
        files = _make_files(15)
        train, val = split_train_val(files, mode="holdout", val_ratio=0.2, seed=7)
        all_imgs = {f["image"] for f in train} | {f["image"] for f in val}
        assert all_imgs == {f["image"] for f in files}

    def test_deterministic_with_same_seed(self):
        files = _make_files(20)
        t1, v1 = split_train_val(files, mode="holdout", val_ratio=0.2, seed=42)
        t2, v2 = split_train_val(files, mode="holdout", val_ratio=0.2, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_seed_gives_different_split(self):
        files = _make_files(20)
        _, v1 = split_train_val(files, mode="holdout", val_ratio=0.2, seed=1)
        _, v2 = split_train_val(files, mode="holdout", val_ratio=0.2, seed=2)
        assert v1 != v2

    def test_val_ratio_0_gives_empty_val(self):
        files = _make_files(10)
        train, val = split_train_val(files, mode="holdout", val_ratio=0.0, seed=42)
        assert len(train) == 10
        assert len(val) == 0

    def test_val_ratio_1_gives_empty_train(self):
        files = _make_files(10)
        train, val = split_train_val(files, mode="holdout", val_ratio=1.0, seed=42)
        assert len(train) == 0
        assert len(val) == 10


class TestSplitKFold:
    def test_fold_sizes_roughly_equal(self):
        files = _make_files(20)
        for fold in range(5):
            train, val = split_train_val(files, mode="kfold", n_folds=5, fold=fold, seed=42)
            assert len(train) + len(val) == 20
            assert len(val) == 4  # 20 / 5

    def test_folds_are_non_overlapping(self):
        files = _make_files(25)
        all_val = []
        for fold in range(5):
            _, val = split_train_val(files, mode="kfold", n_folds=5, fold=fold, seed=42)
            all_val.append({f["image"] for f in val})
        # Check pairwise disjointness
        for i in range(5):
            for j in range(i + 1, 5):
                assert all_val[i] & all_val[j] == set(), f"Folds {i} and {j} overlap"

    def test_folds_cover_all_files(self):
        files = _make_files(25)
        all_val_imgs = set()
        for fold in range(5):
            _, val = split_train_val(files, mode="kfold", n_folds=5, fold=fold, seed=42)
            all_val_imgs |= {f["image"] for f in val}
        assert all_val_imgs == {f["image"] for f in files}

    def test_deterministic_with_same_seed(self):
        files = _make_files(20)
        t1, v1 = split_train_val(files, mode="kfold", n_folds=5, fold=2, seed=42)
        t2, v2 = split_train_val(files, mode="kfold", n_folds=5, fold=2, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_folds_give_different_val_sets(self):
        files = _make_files(20)
        _, v0 = split_train_val(files, mode="kfold", n_folds=5, fold=0, seed=42)
        _, v1 = split_train_val(files, mode="kfold", n_folds=5, fold=1, seed=42)
        assert {f["image"] for f in v0} != {f["image"] for f in v1}

    def test_uneven_split(self):
        """With 23 files and 5 folds, first 3 folds get 5 items, last 2 get 4."""
        files = _make_files(23)
        sizes = []
        for fold in range(5):
            _, val = split_train_val(files, mode="kfold", n_folds=5, fold=fold, seed=42)
            sizes.append(len(val))
        assert sum(sizes) == 23
        assert max(sizes) - min(sizes) <= 1

    def test_2_folds(self):
        files = _make_files(10)
        t0, v0 = split_train_val(files, mode="kfold", n_folds=2, fold=0, seed=42)
        t1, v1 = split_train_val(files, mode="kfold", n_folds=2, fold=1, seed=42)
        assert len(v0) == 5
        assert len(v1) == 5
        assert {f["image"] for f in v0} == {f["image"] for f in t1}


class TestSplitCustom:
    def test_custom_split_from_file(self, tmp_path):
        files = _make_files(6)
        # split_file uses basenames, which must match Path(f["image"]).name
        split_data = {
            "train": [f"img_{i:04d}.png" for i in range(4)],
            "val": [f"img_{i:04d}.png" for i in range(4, 6)],
        }
        split_file = tmp_path / "split.json"
        split_file.write_text(json.dumps(split_data))

        train, val = split_train_val(files, mode="custom", split_file=str(split_file))
        assert len(train) == 4
        assert len(val) == 2
        assert {f["image"] for f in val} == {"/data/img_0004.png", "/data/img_0005.png"}


class TestSplitUnknownMode:
    def test_unknown_mode_returns_all_train(self):
        files = _make_files(5)
        train, val = split_train_val(files, mode="bogus")
        assert train == files
        assert val == []


# ── validate ─────────────────────────────────────────────────────────────────


class TestValidate:
    @pytest.fixture
    def fabric(self):
        mock = MagicMock()
        mock.backward = lambda loss: loss.backward()
        return mock

    @pytest.fixture
    def setup(self):
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
        )
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        dataset = [
            {
                "image": torch.randn(1, 32, 32),
                "label": torch.randint(0, 3, (1, 32, 32)),
            }
            for _ in range(4)
        ]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        return model, loader, loss_fn

    def test_returns_val_loss(self, fabric, setup):
        model, loader, loss_fn = setup
        result = validate(fabric, model, loader, loss_fn)
        assert "val_loss" in result
        assert isinstance(result["val_loss"], float)
        assert result["val_loss"] > 0

    def test_returns_val_metrics(self, fabric, setup):
        model, loader, loss_fn = setup
        result = validate(fabric, model, loader, loss_fn, metrics=["dice", "iou"])
        assert "val_loss" in result
        assert "val_dice" in result
        assert "val_iou" in result
        assert isinstance(result["val_dice"], float)
        assert isinstance(result["val_iou"], float)

    def test_no_metrics(self, fabric, setup):
        model, loader, loss_fn = setup
        result = validate(fabric, model, loader, loss_fn, metrics=None)
        assert "val_loss" in result
        assert "val_dice" not in result

    def test_model_in_train_mode_after_validate(self, fabric, setup):
        model, loader, loss_fn = setup
        model.train()
        validate(fabric, model, loader, loss_fn)
        assert model.training

    def test_no_gradient_accumulation(self, fabric, setup):
        """Validate should not update model weights."""
        model, loader, loss_fn = setup
        initial_params = [p.clone() for p in model.parameters()]
        validate(fabric, model, loader, loss_fn)
        for init, final in zip(initial_params, model.parameters()):
            assert torch.equal(init, final.data)

    def test_cross_entropy_loss(self, fabric):
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
        )
        loss_fn = nn.CrossEntropyLoss()
        dataset = [
            {
                "image": torch.randn(1, 32, 32),
                "label": torch.randint(0, 3, (1, 32, 32)),
            }
            for _ in range(4)
        ]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        result = validate(fabric, model, loader, loss_fn)
        assert result["val_loss"] > 0

    def test_empty_loader(self, fabric):
        """Validate with empty loader should not crash."""
        model = nn.Sequential(nn.Conv2d(1, 3, 1))
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        loader = torch.utils.data.DataLoader([], batch_size=1)
        result = validate(fabric, model, loader, loss_fn)
        # val_loss should be 0.0 / max(0,1) = 0
        assert "val_loss" in result
