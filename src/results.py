from pathlib import Path
from datetime import datetime
import json
import torch


class RunLogger:
    """Manages checkpoint saving for a single training run."""

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        results_root: str = "results",
        resume_from: str = None,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.resume_from = resume_from

        if resume_from:
            # Reuse the original run directory so checkpoints accumulate
            self.run_dir = Path(resume_from)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(results_root) / dataset_name / model_name / timestamp

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        if resume_from:
            print(f"Resuming training from: {resume_from}")
        print(f"Run directory created: {self.run_dir}")

    def save_config(self, config: dict):
        """Save run configuration to the run directory."""
        config_file = self.run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def save_checkpoint(self, model, epoch: int, optimizer=None, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()

        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
        else:
            print(f"Checkpoint saved to {checkpoint_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> dict:
        """Load a checkpoint file."""
        checkpoint = torch.load(checkpoint_path)
        return {
            "epoch": checkpoint.get("epoch", 0),
            "model_state": checkpoint.get("model_state", {}),
            "optimizer_state": checkpoint.get("optimizer_state", None),
        }

    @staticmethod
    def load_run_config(run_path: str) -> dict:
        """Load config from a previous run."""
        config_file = Path(run_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file) as f:
            return json.load(f)

    @staticmethod
    def get_checkpoint_path(run_path: str, checkpoint_name: str = "best_model.pt") -> Path:
        """Get the full path to a checkpoint file."""
        checkpoint_path = Path(run_path) / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path
