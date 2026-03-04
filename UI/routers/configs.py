"""Saved configs management API routes."""

import json
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Directory to store saved configs
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)


def _reset_stale_running():
    """On startup, reset any configs stuck in 'running' back to 'idle'."""
    for config_file in CONFIGS_DIR.glob("*.json"):
        try:
            with open(config_file) as f:
                data = json.load(f)
            if data.get("status") == "running":
                data["status"] = "idle"
                with open(config_file, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass


_reset_stale_running()


class ConfigRequest(BaseModel):
    """Request model for saving a config."""

    name: str
    command: str
    params: dict = {}


def get_config_path(name: str) -> Path:
    """Get path for a config file."""
    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    return CONFIGS_DIR / f"{safe_name}.json"


_EPOCH_RE = re.compile(r"epoch_(\d+)\.pt$")


def _get_checkpoint_epoch(config_data: dict) -> int:
    """Return the highest completed epoch from a config's run_dir, or 0."""
    run_dir = config_data.get("run_dir")
    if not run_dir:
        return 0
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return 0
    max_epoch = 0
    for f in ckpt_dir.iterdir():
        m = _EPOCH_RE.search(f.name)
        if m:
            max_epoch = max(max_epoch, int(m.group(1)))
    return max_epoch


@router.post("/api/configs/save")
async def save_config(config: ConfigRequest):
    """Save a new config."""
    try:
        # Ensure configs directory exists
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

        path = get_config_path(config.name)

        config_data = {
            "name": config.name,
            "command": config.command,
            "params": config.params,
        }

        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)

        return {"success": True, "name": config.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@router.get("/api/configs/list")
async def list_configs():
    """List all saved configs."""
    configs = []

    if not CONFIGS_DIR.exists():
        return configs

    for config_file in sorted(CONFIGS_DIR.glob("*.json")):
        try:
            with open(config_file) as f:
                config_data = json.load(f)
                config_data["checkpoint_epoch"] = _get_checkpoint_epoch(config_data)
                configs.append(config_data)
        except Exception as e:
            print(f"Error reading config {config_file}: {e}")

    return configs


@router.get("/api/configs/get/{config_name}")
async def get_config(config_name: str):
    """Get a specific config."""
    path = get_config_path(config_name)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Config not found")

    try:
        with open(path) as f:
            config_data = json.load(f)
        return config_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {e}")


@router.patch("/api/configs/status/{config_name}")
async def update_config_status(config_name: str, status: str):
    """Update the status field of a config (idle, running, done)."""
    path = get_config_path(config_name)
    if not path.exists():
        return
    try:
        with open(path) as f:
            config_data = json.load(f)
        config_data["status"] = status
        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)
    except Exception:
        pass


def set_config_status(config_name: str, status: str):
    """Set config status synchronously (for use from drain thread)."""
    set_config_field(config_name, "status", status)


def set_config_field(config_name: str, key: str, value):
    """Set an arbitrary field on a config JSON file."""
    path = get_config_path(config_name)
    if not path.exists():
        return
    try:
        with open(path) as f:
            config_data = json.load(f)
        config_data[key] = value
        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)
    except Exception:
        pass


@router.delete("/api/configs/delete/{config_name}")
async def delete_config(config_name: str):
    """Delete a config."""
    path = get_config_path(config_name)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Config not found")

    try:
        path.unlink()
        return {"success": True, "name": config_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting config: {e}")
