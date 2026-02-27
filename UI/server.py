#!/usr/bin/env python3
"""FastAPI server for MonaiUI web interface and API."""

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import DATASETS, MODELS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="MonaiUI",
    description="Medical image segmentation framework",
    version="0.1.0",
)


@app.get("/api/models")
async def get_models():
    """Get available models."""
    return MODELS


@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets."""
    return DATASETS


@app.get("/api/results")
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

                if not config_file.exists() or not summary_file.exists():
                    continue

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
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error loading result from {run_dir}: {e}")
                    continue

    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results


@app.get("/")
async def root():
    """Serve index.html."""
    index_path = Path(__file__).parent / "gui" / "index.html"
    with open(index_path) as f:
        return HTMLResponse(content=f.read())


# Mount static files (CSS, JS, etc.)
gui_path = Path(__file__).parent / "gui"
if gui_path.exists():
    app.mount("/static", StaticFiles(directory=str(gui_path)), name="static")


if __name__ == "__main__":
    import uvicorn

    PORT = 8888
    print("Starting MonaiUI...")
    print(f"🌐 Web UI: http://localhost:{PORT}")
    print(f"📚 API Docs: http://localhost:{PORT}/docs")
    print(f"🔌 OpenAPI Schema: http://localhost:{PORT}/openapi.json")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
