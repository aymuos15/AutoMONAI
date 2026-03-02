"""Saved configs management API routes."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Directory to store saved configs
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)


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
