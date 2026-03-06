"""Tests for the new CLI arguments (val_split, best_metric, ensemble)."""

import pytest

from automonai.core.cli import get_parser
from automonai.core.config import VAL_SPLIT_MODES, BEST_METRIC_CHOICES, ENSEMBLE_METHODS


class TestValSplitArgs:
    def test_default_val_split_is_none(self):
        args = get_parser().parse_args([])
        assert args.val_split == "none"

    @pytest.mark.parametrize("mode", VAL_SPLIT_MODES)
    def test_accepts_all_val_split_modes(self, mode):
        args = get_parser().parse_args(["--val_split", mode])
        assert args.val_split == mode

    def test_rejects_invalid_val_split(self):
        with pytest.raises(SystemExit):
            get_parser().parse_args(["--val_split", "bogus"])

    def test_default_val_ratio(self):
        args = get_parser().parse_args([])
        assert args.val_ratio == 0.2

    def test_custom_val_ratio(self):
        args = get_parser().parse_args(["--val_ratio", "0.3"])
        assert args.val_ratio == 0.3

    def test_default_split_seed(self):
        args = get_parser().parse_args([])
        assert args.split_seed == 42

    def test_custom_split_seed(self):
        args = get_parser().parse_args(["--split_seed", "123"])
        assert args.split_seed == 123

    def test_default_split_file_is_none(self):
        args = get_parser().parse_args([])
        assert args.split_file is None

    def test_custom_split_file(self):
        args = get_parser().parse_args(["--split_file", "/tmp/split.json"])
        assert args.split_file == "/tmp/split.json"


class TestBestMetricArg:
    def test_default_is_val_loss(self):
        args = get_parser().parse_args([])
        assert args.best_metric == "val_loss"

    @pytest.mark.parametrize("metric", BEST_METRIC_CHOICES)
    def test_accepts_all_best_metric_choices(self, metric):
        args = get_parser().parse_args(["--best_metric", metric])
        assert args.best_metric == metric

    def test_rejects_invalid_best_metric(self):
        with pytest.raises(SystemExit):
            get_parser().parse_args(["--best_metric", "invalid"])


class TestEnsembleArgs:
    def test_ensemble_folds_default_false(self):
        args = get_parser().parse_args([])
        assert args.ensemble_folds is False

    def test_ensemble_folds_flag(self):
        args = get_parser().parse_args(["--ensemble_folds"])
        assert args.ensemble_folds is True

    def test_ensemble_method_default(self):
        args = get_parser().parse_args([])
        assert args.ensemble_method == "mean"

    @pytest.mark.parametrize("method", ENSEMBLE_METHODS)
    def test_accepts_all_ensemble_methods(self, method):
        args = get_parser().parse_args(["--ensemble_method", method])
        assert args.ensemble_method == method

    def test_rejects_invalid_ensemble_method(self):
        with pytest.raises(SystemExit):
            get_parser().parse_args(["--ensemble_method", "invalid"])

    def test_fold_dirs_default_empty(self):
        args = get_parser().parse_args([])
        assert args.fold_dirs == []

    def test_fold_dirs_multiple(self):
        args = get_parser().parse_args(["--fold_dirs", "/run/fold1", "/run/fold2", "/run/fold3"])
        assert args.fold_dirs == ["/run/fold1", "/run/fold2", "/run/fold3"]


class TestCombinedArgs:
    def test_kfold_with_val_split(self):
        args = get_parser().parse_args([
            "--cross_val", "5", "--cv_fold", "2",
            "--val_split", "kfold", "--split_seed", "99",
        ])
        assert args.cross_val == 5
        assert args.cv_fold == 2
        assert args.val_split == "kfold"
        assert args.split_seed == 99

    def test_holdout_with_best_metric(self):
        args = get_parser().parse_args([
            "--val_split", "holdout", "--val_ratio", "0.15",
            "--best_metric", "val_dice",
        ])
        assert args.val_split == "holdout"
        assert args.val_ratio == 0.15
        assert args.best_metric == "val_dice"

    def test_ensemble_with_fold_dirs(self):
        args = get_parser().parse_args([
            "--ensemble_folds", "--ensemble_method", "vote",
            "--fold_dirs", "/a", "/b",
        ])
        assert args.ensemble_folds is True
        assert args.ensemble_method == "vote"
        assert args.fold_dirs == ["/a", "/b"]
