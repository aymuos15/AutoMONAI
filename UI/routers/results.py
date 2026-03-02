"""Training results API routes."""

import json
import shutil
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/results")
async def get_results():
    """Get all training results from the results directory."""
    results = []
    results_dir = Path("results")

    if not results_dir.exists():
        return results

    # Iterate through results/dataset/model/timestamp structure
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # Load result data
                config_file = run_dir / "config.json"
                summary_file = run_dir / "summary.json"
                metrics_file = run_dir / "metrics.csv"

                status = "in_progress"
                if summary_file.exists():
                    status = "complete"

                try:
                    with open(config_file) as f:
                        config = json.load(f)

                    with open(summary_file) as f:
                        summary = json.load(f)

                    # Parse metrics CSV
                    metrics = []
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                headers = lines[0].strip().split(",")
                                for line in lines[1:]:
                                    values = line.strip().split(",")
                                    metric_dict = {}
                                    for h, v in zip(headers, values):
                                        try:
                                            metric_dict[h] = float(v)
                                        except ValueError:
                                            metric_dict[h] = v
                                    metrics.append(metric_dict)

                    result = {
                        "dataset": dataset_dir.name,
                        "model": model_dir.name,
                        "timestamp": run_dir.name,
                        "config": config,
                        "summary": summary,
                        "metrics": metrics,
                        "epochs": summary.get("total_epochs", 0),
                        "best_loss": summary.get("best_loss", 0),
                        "status": status,
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error loading result from {run_dir}: {e}")
                    continue

    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results


@router.delete("/api/results/{dataset}/{model}/{timestamp}")
async def delete_result(dataset: str, model: str, timestamp: str):
    """Delete a training result."""
    try:
        run_dir = Path("results") / dataset / model / timestamp
        if not run_dir.exists():
            return {"error": "Result not found"}, 404

        shutil.rmtree(run_dir)
        return {"status": "deleted"}
    except Exception as e:
        return {"error": str(e)}, 500
