#!/usr/bin/env python3
"""FastAPI server for MonaiUI web interface and API."""

import asyncio
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import DATASETS, MODELS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Launch state
_proc: subprocess.Popen | None = None
_log_buffer: list[str] = []


class LaunchRequest(BaseModel):
    command: str

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


def _drain(proc: subprocess.Popen) -> None:
    """Read stdout into log buffer (blocking, run in thread)."""
    try:
        for line in proc.stdout:
            _log_buffer.append(line.rstrip())
    except Exception as e:
        print(f"Error reading process output: {e}")
    finally:
        proc.wait()


@app.post("/api/launch")
async def launch_training(req: LaunchRequest):
    """Launch training with the given command."""
    global _proc, _log_buffer

    # Validate command starts with python3 -m src.run
    if not req.command.startswith("python3 -m src.run"):
        return {"error": "Invalid command"}, 400

    # Check if already running
    if _proc is not None and _proc.poll() is None:
        return {"error": "Training already running"}, 409

    # Clear log buffer
    _log_buffer.clear()

    try:
        # Spawn process with unbuffered output
        _proc = subprocess.Popen(
            ["python3", "-u", "-m", "src.run"] + shlex.split(req.command.replace("python3 -m src.run ", "")),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Fire drain thread (non-blocking)
        drain_thread = threading.Thread(target=_drain, args=(_proc,), daemon=True)
        drain_thread.start()

        return {"status": "launched"}
    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/api/launch/status")
async def launch_status():
    """Get launch status."""
    if _proc is None:
        return {"running": False, "returncode": None}

    returncode = _proc.poll()
    return {"running": returncode is None, "returncode": returncode}


@app.get("/api/launch/logs")
async def launch_logs():
    """Stream logs via SSE."""
    async def event_generator():
        tail_index = [0]  # Use a list to allow modification in nested scope

        while True:
            # Send any new lines since last tail index
            while tail_index[0] < len(_log_buffer):
                line = _log_buffer[tail_index[0]]
                tail_index[0] += 1
                yield f"data: {line}\n\n"

            # Check if process is done
            if _proc is not None and _proc.poll() is not None:
                # Process exited, send any remaining lines
                while tail_index[0] < len(_log_buffer):
                    line = _log_buffer[tail_index[0]]
                    tail_index[0] += 1
                    yield f"data: {line}\n\n"

                # Send done event
                yield "event: done\ndata: \n\n"
                break

            # Sleep briefly before polling again
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/launch/stop")
async def launch_stop():
    """Stop the running training."""
    global _proc

    if _proc is not None and _proc.poll() is None:
        try:
            _proc.terminate()
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
        except Exception as e:
            return {"error": str(e)}, 500

    return {"status": "stopped"}


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


@app.delete("/api/results/{dataset}/{model}/{timestamp}")
async def delete_result(dataset: str, model: str, timestamp: str):
    """Delete a training result."""
    try:
        import shutil

        run_dir = Path("results") / dataset / model / timestamp
        if not run_dir.exists():
            return {"error": "Result not found"}, 404

        shutil.rmtree(run_dir)
        return {"status": "deleted"}
    except Exception as e:
        return {"error": str(e)}, 500


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
