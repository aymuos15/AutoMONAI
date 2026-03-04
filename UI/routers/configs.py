"""Saved configs management API routes."""

import json
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

def _import_wandb():
    """Import the real wandb package, bypassing the local wandb/ log directory."""
    import importlib
    import sys
    project_root = str(Path(__file__).parent.parent.parent)
    saved = sys.path[:]
    try:
        # Remove project root and '' (cwd) so the local wandb/ dir isn't found
        sys.path = [p for p in sys.path if p not in (project_root, "", ".")]
        for key in list(sys.modules):
            if key == "wandb" or key.startswith("wandb."):
                del sys.modules[key]
        mod = importlib.import_module("wandb")
        if not hasattr(mod, "Api"):
            raise ImportError
        return mod
    except ImportError:
        return None
    finally:
        sys.path = saved


wandb = _import_wandb()

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


@router.post("/api/configs/sync-wandb")
async def sync_wandb():
    """Sync W&B runs with local configs: delete orphans, update changed configs."""
    if wandb is None:
        raise HTTPException(status_code=500, detail="wandb is not installed")

    try:
        api = wandb.Api()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to W&B: {e}")

    # Load local config names
    local_names = set()
    local_configs = {}
    if CONFIGS_DIR.exists():
        for config_file in CONFIGS_DIR.glob("*.json"):
            try:
                with open(config_file) as f:
                    data = json.load(f)
                name = data.get("name", config_file.stem)
                local_names.add(name)
                local_configs[name] = data
            except Exception:
                pass

    deleted = []
    updated = []
    unchanged = []

    try:
        runs = api.runs("AutoMONAI")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list W&B runs: {e}"
        )

    for run in runs:
        run_id = run.id
        run_name = run.name

        # Match by run name (which equals config name)
        if run_name not in local_names:
            try:
                run.delete()
                deleted.append(run_name or run_id)
            except Exception as e:
                print(f"Failed to delete W&B run {run_id}: {e}")
        else:
            # Check if local config changed — update W&B run config
            local_data = local_configs.get(run_name, {})
            local_params = local_data.get("params", {})
            wandb_config = dict(run.config) if run.config else {}

            if local_params and local_params != wandb_config:
                try:
                    run.config.update(local_params)
                    run.update()
                    updated.append(run_name)
                except Exception as e:
                    print(f"Failed to update W&B run {run_id}: {e}")
            else:
                unchanged.append(run_name)

    return {"deleted": deleted, "updated": updated, "unchanged": unchanged}


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
