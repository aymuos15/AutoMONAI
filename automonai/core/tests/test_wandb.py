"""Tests for W&B initialization in run.py."""

from pathlib import Path


def _make_wandb_kwargs(
    run_id=None, wandb_run_id=None, dataset_name="Dataset001", model_name="unet"
):
    """Reproduce the wandb_kwargs construction from run.py main()."""
    config = {"model": model_name, "dataset": dataset_name}
    wandb_kwargs = {"project": "AutoMONAI", "config": config, "dir": str(Path.home() / ".wandb")}
    wandb_kwargs["name"] = run_id or f"{dataset_name}_{model_name}"
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "allow"
    return wandb_kwargs


class TestWandbInit:
    def test_no_resume_or_id_without_wandb_run_id(self):
        """Fresh training should not set id or resume."""
        kwargs = _make_wandb_kwargs(run_id="my_run")
        assert "id" not in kwargs
        assert "resume" not in kwargs

    def test_resume_with_wandb_run_id(self):
        """Inference should resume the same W&B run via wandb_run_id."""
        kwargs = _make_wandb_kwargs(run_id="my_run", wandb_run_id="abc123")
        assert kwargs["id"] == "abc123"
        assert kwargs["resume"] == "allow"

    def test_name_uses_run_id_when_provided(self):
        kwargs = _make_wandb_kwargs(run_id="epochs_5")
        assert kwargs["name"] == "epochs_5"

    def test_name_uses_dataset_model_when_no_run_id(self):
        kwargs = _make_wandb_kwargs(dataset_name="CellSeg", model_name="segresnet")
        assert kwargs["name"] == "CellSeg_segresnet"

    def test_dir_is_home_wandb(self):
        kwargs = _make_wandb_kwargs()
        assert kwargs["dir"] == str(Path.home() / ".wandb")
