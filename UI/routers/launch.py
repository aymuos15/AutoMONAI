"""Training launch API routes with live log streaming."""

import asyncio
import json
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from UI.routers.configs import get_config_path, set_config_status, set_config_field

router = APIRouter()

# Multi-process registry: run_id -> {proc, log_buffer, finished_at}
_processes: dict[str, dict] = {}
_lock = threading.Lock()

_CLEANUP_SECONDS = 30


class LaunchRequest(BaseModel):
    command: str
    run_id: str = "__main__"


class StopRequest(BaseModel):
    run_id: str = "__main__"


def _get_run(run_id: str) -> Optional[dict]:
    """Get a run entry, cleaning up if expired."""
    with _lock:
        entry = _processes.get(run_id)
        if entry is None:
            return None
        # Clean up finished processes after timeout
        if entry.get("finished_at") and time.time() - entry["finished_at"] > _CLEANUP_SECONDS:
            del _processes[run_id]
            return None
        return entry


_RUN_DIR_RE = re.compile(r"Run directory created: (.+)")


def _find_resume_checkpoint(run_id: str) -> tuple[Optional[str], Optional[str]]:
    """Find the best checkpoint to resume from using the config's stored run_dir."""
    cfg_path = get_config_path(run_id)
    if not cfg_path.exists():
        return None, None
    with open(cfg_path) as f:
        cfg_data = json.load(f)
    for key in ("run_dir", "original_run_dir"):
        run_dir = cfg_data.get(key)
        if run_dir:
            result = _find_latest_checkpoint(run_dir)
            if result[0]:
                return result
    return None, None


def _find_latest_checkpoint(run_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Find the latest checkpoint in a run dir.

    Prefers the latest epoch checkpoint (most recent training state).
    Falls back to best_model.pt if no epoch checkpoints exist.
    Returns (run_dir, checkpoint_filename) or (None, None) if no checkpoint found.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None, None

    # Prefer latest epoch checkpoint (most recent training state)
    epoch_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if epoch_files:
        return run_dir, epoch_files[-1].name

    # Fall back to best_model.pt
    if (ckpt_dir / "best_model.pt").exists():
        return run_dir, "best_model.pt"

    return None, None


def _drain(proc: subprocess.Popen, log_buffer: list[str], run_id: str) -> None:
    """Read stdout into log buffer (blocking, run in thread)."""
    try:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            stripped = line.rstrip()
            log_buffer.append(stripped)
            # Capture run directory for resume support
            if run_id != "__main__":
                m = _RUN_DIR_RE.search(stripped)
                if m:
                    new_dir = m.group(1)
                    # Preserve original_run_dir for checkpoint chain
                    with open(get_config_path(run_id)) as f:
                        cfg = json.load(f)
                    if not cfg.get("original_run_dir"):
                        set_config_field(run_id, "original_run_dir", new_dir)
                    set_config_field(run_id, "run_dir", new_dir)
    except Exception as e:
        print(f"Error reading process output ({run_id}): {e}")
    finally:
        proc.wait()
        with _lock:
            entry = _processes.get(run_id)
            if entry:
                entry["finished_at"] = time.time()
        if run_id != "__main__":
            set_config_status(run_id, "done" if proc.returncode == 0 else "idle")


@router.post("/api/launch")
async def launch_training(req: LaunchRequest):
    """Launch training with the given command."""
    if not req.command.startswith("python3 -m src.run"):
        raise HTTPException(status_code=400, detail="Invalid command")

    existing = _get_run(req.run_id)
    if existing and existing["proc"].poll() is None:
        raise HTTPException(status_code=409, detail="Training already running")

    log_buffer: list[str] = []

    # Append --run_id so W&B reuses the same run on re-launch
    extra_args = []
    if req.run_id != "__main__":
        extra_args = ["--run_id", req.run_id]

        # Auto-resume from last checkpoint if available
        resume_dir, ckpt_name = _find_resume_checkpoint(req.run_id)
        if resume_dir:
            extra_args += ["--resume", resume_dir, "--checkpoint", ckpt_name]

    try:
        proc = subprocess.Popen(
            ["python3", "-u", "-m", "src.run"]
            + shlex.split(req.command.replace("python3 -m src.run ", ""))
            + extra_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with _lock:
            _processes[req.run_id] = {
                "proc": proc,
                "log_buffer": log_buffer,
                "finished_at": None,
            }

        drain_thread = threading.Thread(
            target=_drain, args=(proc, log_buffer, req.run_id), daemon=True
        )
        drain_thread.start()

        if req.run_id != "__main__":
            set_config_status(req.run_id, "running")
        return {"status": "launched", "run_id": req.run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/launch/status")
async def launch_status(run_id: str = Query("__main__")):
    """Get launch status for a specific run."""
    entry = _get_run(run_id)
    if entry is None:
        return {"running": False, "returncode": None}

    returncode = entry["proc"].poll()
    return {"running": returncode is None, "returncode": returncode}


@router.get("/api/launch/logs")
async def launch_logs(run_id: str = Query("__main__")):
    """Stream logs via SSE for a specific run."""

    async def event_generator():
        tail_index = [0]

        while True:
            current = _get_run(run_id)
            if current is None:
                yield "event: done\ndata: \n\n"
                break

            log_buffer = current["log_buffer"]
            proc = current["proc"]

            while tail_index[0] < len(log_buffer):
                line = log_buffer[tail_index[0]]
                tail_index[0] += 1
                yield f"data: {line}\n\n"

            if proc.poll() is not None:
                # Flush remaining
                while tail_index[0] < len(log_buffer):
                    line = log_buffer[tail_index[0]]
                    tail_index[0] += 1
                    yield f"data: {line}\n\n"
                yield "event: done\ndata: \n\n"
                break

            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/launch/stop")
async def launch_stop(req: Optional[StopRequest] = None):
    """Stop a running training."""
    run_id = req.run_id if req else "__main__"

    entry = _get_run(run_id)
    if entry and entry["proc"].poll() is None:
        try:
            entry["proc"].terminate()
            entry["proc"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            entry["proc"].kill()

    if run_id != "__main__":
        set_config_status(run_id, "idle")

    return {"status": "stopped"}


@router.get("/api/launch/list")
async def launch_list():
    """Return all active/recent run states."""
    result = {}
    with _lock:
        to_delete = []
        for run_id, entry in _processes.items():
            if entry.get("finished_at") and time.time() - entry["finished_at"] > _CLEANUP_SECONDS:
                to_delete.append(run_id)
                continue
            returncode = entry["proc"].poll()
            result[run_id] = {
                "running": returncode is None,
                "returncode": returncode,
            }
        for rid in to_delete:
            del _processes[rid]
    return result
