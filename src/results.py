import json
import csv
from pathlib import Path
from datetime import datetime
import torch


class RunLogger:
    """Manages logging for a single training run."""

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        results_root: str = "results",
        resume_from: str = None,
    ):
        """Initialize run logger with unique timestamp-based directory.

        Args:
            dataset_name: Name of the dataset used
            model_name: Name of the model used
            results_root: Root directory for all results (default: "results")
            resume_from: Optional path to existing run to resume from (creates new timestamp)
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.resume_from = resume_from

        if resume_from:
            resume_path = Path(resume_from)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = (
                Path(results_root)
                / resume_path.parent.parent.name
                / resume_path.parent.name
                / f"{resume_path.name}_resumed_{timestamp}"
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(results_root) / dataset_name / model_name / timestamp

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.config_file = self.run_dir / "config.json"
        self.metrics_file = self.run_dir / "metrics.csv"
        self.log_file = self.run_dir / "training.log"

        self.metrics_history = []
        self.csv_writer = None
        self.csv_file = None

        if resume_from:
            print(f"Resuming training from: {resume_from}")
        print(f"Run directory created: {self.run_dir}")

    def save_config(self, config: dict):
        """Save training configuration to JSON.

        Args:
            config: Dictionary of configuration parameters
        """
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {self.config_file}")

    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an epoch to CSV and memory.

        Args:
            epoch: Epoch number (0-indexed)
            metrics: Dictionary of metric values (e.g., {"loss": 0.5, "dice": 0.8})
        """
        # Initialize CSV on first call
        if self.csv_file is None:
            self.csv_file = open(self.metrics_file, "w", newline="")
            fieldnames = ["epoch"] + list(metrics.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()

        # Write row
        row = {"epoch": epoch + 1}  # 1-indexed for readability
        row.update(metrics)
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # Keep in memory
        self.metrics_history.append({"epoch": epoch + 1, **metrics})

    def save_checkpoint(self, model, epoch: int, optimizer=None, is_best: bool = False):
        """Save model checkpoint.

        Args:
            model: PyTorch model or wrapped model
            epoch: Epoch number (0-indexed)
            optimizer: Optional optimizer state
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
        else:
            print(f"Checkpoint saved to {checkpoint_path}")

    def save_final_summary(self, summary: dict):
        """Save final training summary to JSON.

        Args:
            summary: Dictionary with final metrics and stats
        """
        summary_file = self.run_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

    def get_metrics_json(self) -> dict:
        """Get all metrics as a dictionary."""
        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "run_dir": str(self.run_dir),
            "metrics": self.metrics_history,
        }

    def close(self):
        """Close CSV file handle."""
        if self.csv_file is not None:
            self.csv_file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> dict:
        """Load a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt)

        Returns:
            Dictionary with 'epoch', 'model_state', 'optimizer_state' (if saved)
        """
        checkpoint = torch.load(checkpoint_path)
        return {
            "epoch": checkpoint.get("epoch", 0),
            "model_state": checkpoint.get("model_state", {}),
            "optimizer_state": checkpoint.get("optimizer_state", None),
        }

    @staticmethod
    def load_run_config(run_path: str) -> dict:
        """Load config from a previous run.

        Args:
            run_path: Path to the run directory

        Returns:
            Dictionary with configuration parameters
        """
        config_file = Path(run_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file) as f:
            return json.load(f)

    @staticmethod
    def get_checkpoint_path(run_path: str, checkpoint_name: str = "best_model.pt") -> Path:
        """Get the full path to a checkpoint file.

        Args:
            run_path: Path to the run directory
            checkpoint_name: Name of checkpoint file (default: best_model.pt)

        Returns:
            Path object to the checkpoint
        """
        checkpoint_path = Path(run_path) / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
