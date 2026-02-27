"""Training launch API routes with live log streaming."""

import asyncio
import shlex
import subprocess
import threading
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# Launch state
_proc: Optional[subprocess.Popen] = None
_log_buffer: list[str] = []


class LaunchRequest(BaseModel):
    command: str


def _drain(proc: subprocess.Popen) -> None:
    """Read stdout into log buffer (blocking, run in thread)."""
    try:
        for line in proc.stdout:
            _log_buffer.append(line.rstrip())
    except Exception as e:
        print(f"Error reading process output: {e}")
    finally:
        proc.wait()


@router.post("/api/launch")
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


@router.get("/api/launch/status")
async def launch_status():
    """Get launch status."""
    if _proc is None:
        return {"running": False, "returncode": None}

    returncode = _proc.poll()
    return {"running": returncode is None, "returncode": returncode}


@router.get("/api/launch/logs")
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


@router.post("/api/launch/stop")
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
