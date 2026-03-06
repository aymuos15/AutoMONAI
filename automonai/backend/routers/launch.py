"""Training launch API routes with live log streaming."""

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from automonai.backend.routers.configs import (
    get_config_path,
    set_config_status,
    set_config_field,
    set_fold_field,
    set_fold_status,
    get_fold_state,
)

router = APIRouter()

# Project root (3 levels up from this file: routers/ -> backend/ -> automonai/ -> project root)
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)

# Multi-process registry: run_id -> {proc, log_buffer, finished_at}
_processes: dict[str, dict] = {}
_lock = threading.Lock()

_CLEANUP_SECONDS = 30


class LaunchRequest(BaseModel):
    command: str
    run_id: str = "__main__"
    variant_id: Optional[str] = None


class StopRequest(BaseModel):
    run_id: str = "__main__"
    variant_id: Optional[str] = None


def _effective_run_id(run_id: str, variant_id: Optional[str]) -> str:
    """Build process registry run ID for variant launches."""
    if variant_id and variant_id != "no_val" and run_id != "__main__":
        return f"{run_id}__{variant_id}"
    return run_id


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
_WANDB_ID_RE = re.compile(r"W&B run ID: (.+)")


def _find_resume_checkpoint(run_id: str, variant_id: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Find the best checkpoint to resume from using the fold's stored run_dir."""
    cfg_path = get_config_path(run_id)
    if not cfg_path.exists():
        return None, None
    with open(cfg_path) as f:
        cfg_data = json.load(f)

    # Check fold-specific state first
    vid = variant_id or "no_val"
    fold_data = cfg_data.get("fold_state", {}).get(vid, {})
    for key in ("run_dir", "original_run_dir"):
        run_dir = fold_data.get(key)
        if run_dir:
            result = _find_latest_checkpoint(run_dir)
            if result[0]:
                return result

    # Fall back to top-level only for no_val (legacy configs)
    if vid == "no_val":
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


def _drain(
    proc: subprocess.Popen,
    log_buffer: list[str],
    run_id: str,
    config_name: Optional[str] = None,
    variant_id: Optional[str] = None,
    is_infer: bool = False,
) -> None:
    """Read stdout into log buffer (blocking, run in thread)."""
    vid = variant_id or "no_val"
    try:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            stripped = line.rstrip()
            log_buffer.append(stripped)
            # Capture run directory and wandb ID per fold
            if config_name and config_name != "__main__":
                wm = _WANDB_ID_RE.search(stripped)
                if wm:
                    set_fold_field(config_name, vid, "wandb_run_id", wm.group(1))
                m = _RUN_DIR_RE.search(stripped)
                if m:
                    new_dir = m.group(1)
                    fold_data = get_fold_state(config_name, vid)
                    if not fold_data.get("original_run_dir"):
                        set_fold_field(config_name, vid, "original_run_dir", new_dir)
                    set_fold_field(config_name, vid, "run_dir", new_dir)
    except Exception as e:
        print(f"Error reading process output ({run_id}): {e}")
    finally:
        proc.wait()
        with _lock:
            entry = _processes.get(run_id)
            if entry:
                entry["finished_at"] = time.time()
        if config_name and config_name != "__main__":
            final_status = "idle"
            if proc.returncode == 0:
                final_status = "inferred" if is_infer else "done"
            set_fold_status(config_name, vid, final_status)


@router.post("/api/launch")
async def launch_training(req: LaunchRequest):
    """Launch training with the given command."""
    if not req.command.startswith("python3 -m automonai.core.run"):
        raise HTTPException(status_code=400, detail="Invalid command")

    effective_run_id = _effective_run_id(req.run_id, req.variant_id)

    existing = _get_run(effective_run_id)
    if existing and existing["proc"].poll() is None:
        raise HTTPException(status_code=409, detail="Training already running")

    log_buffer: list[str] = []

    extra_args = []
    if req.run_id != "__main__":
        extra_args = ["--run_id", effective_run_id]

        # Pass W&B run ID so inference logs to the same W&B run as training
        vid = req.variant_id or "no_val"
        fold_data = get_fold_state(req.run_id, vid)
        wandb_id = fold_data.get("wandb_run_id")
        if wandb_id:
            extra_args += ["--wandb_run_id", wandb_id]

        # Ensemble variant: collect fold run_dirs and inject --fold_dirs
        if vid == "ensemble":
            cfg_path = get_config_path(req.run_id)
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg_data = json.load(f)
                fold_state = cfg_data.get("fold_state", {})
                fold_dirs = []
                for fid, fstate in sorted(fold_state.items()):
                    if fid.startswith("fold_") and fstate.get("run_dir"):
                        fold_dirs.append(fstate["run_dir"])
                if fold_dirs:
                    extra_args += ["--fold_dirs"] + fold_dirs
        else:
            # Auto-resume from last checkpoint if training was incomplete
            resume_dir, ckpt_name = _find_resume_checkpoint(req.run_id, req.variant_id)
            if resume_dir and ckpt_name:
                # Parse total epochs from command and checkpoint epoch
                cmd_epochs_match = re.search(r"--epochs\s+(\d+)", req.command)
                total_epochs = int(cmd_epochs_match.group(1)) if cmd_epochs_match else 0
                ckpt_epoch = int(re.search(r"(\d+)", ckpt_name).group(1)) if re.search(r"(\d+)", ckpt_name) else 0
                if ckpt_epoch < total_epochs:
                    extra_args += ["--resume", resume_dir, "--checkpoint", ckpt_name]

    try:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "automonai.core.run"]
            + shlex.split(req.command.replace("python3 -m automonai.core.run ", ""))
            + extra_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=_PROJECT_ROOT,
            env=env,
        )

        with _lock:
            _processes[effective_run_id] = {
                "proc": proc,
                "log_buffer": log_buffer,
                "finished_at": None,
            }

        is_infer = "--mode infer" in req.command or "--mode=infer" in req.command or "--ensemble_folds" in req.command
        drain_thread = threading.Thread(
            target=_drain,
            args=(proc, log_buffer, effective_run_id, req.run_id, req.variant_id, is_infer),
            daemon=True,
        )
        drain_thread.start()

        if req.run_id != "__main__":
            set_fold_status(req.run_id, req.variant_id or "no_val", "running")
        return {"status": "launched", "run_id": effective_run_id}
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

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/launch/stop")
async def launch_stop(req: Optional[StopRequest] = None):
    """Stop a running training."""
    run_id = req.run_id if req else "__main__"
    variant_id = req.variant_id if req else None
    effective_run_id = _effective_run_id(run_id, variant_id)

    entry = _get_run(effective_run_id)
    if entry and entry["proc"].poll() is None:
        try:
            entry["proc"].terminate()
            entry["proc"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            entry["proc"].kill()

    if run_id != "__main__":
        set_fold_status(run_id, variant_id or "no_val", "idle")

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
