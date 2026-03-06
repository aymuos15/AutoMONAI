"""Saved configs management API routes."""

import json
import re
import threading
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
DEFAULT_FOLD_COUNT = 5

# Lock to prevent concurrent config file reads/writes from racing
_config_lock = threading.Lock()


def _reset_stale_running():
    """On startup, reset any configs stuck in 'running' back to 'idle'."""
    for config_file in CONFIGS_DIR.glob("*.json"):
        with open(config_file) as f:
            data = json.load(f)
        normalized = _normalize_config_schema(data)
        changed = normalized != data
        # Reset top-level status
        if normalized.get("status") == "running":
            normalized["status"] = "idle"
            changed = True
        # Reset per-fold statuses
        fold_state = normalized.get("fold_state", {})
        for fold_id, fstate in fold_state.items():
            if fstate.get("status") == "running":
                fstate["status"] = "idle"
                changed = True
        if changed:
            with open(config_file, "w") as f:
                json.dump(normalized, f, indent=2)


class ConfigRequest(BaseModel):
    """Request model for saving a config."""

    name: str
    command: str
    params: dict = {}
    cv: dict = {}


def get_config_path(name: str) -> Path:
    """Get path for a config file."""
    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    return CONFIGS_DIR / f"{safe_name}.json"


_EPOCH_RE = re.compile(r"epoch_(\d+)\.pt$")
_CROSS_VAL_RE = re.compile(r"\s--cross_val(?:\s+|=)\d+")
_CV_FOLD_RE = re.compile(r"\s--cv_fold(?:\s+|=)\d+")
_VAL_SPLIT_RE = re.compile(r"\s--val_split(?:\s+|=)\S+")
_ENSEMBLE_FOLDS_RE = re.compile(r"\s--ensemble_folds")
_ENSEMBLE_METHOD_RE = re.compile(r"\s--ensemble_method(?:\s+|=)\S+")
_FOLD_DIRS_RE = re.compile(r"\s--fold_dirs(?:\s+\S+)*")


def _strip_cv_flags(command: str) -> str:
    """Strip cross-validation and val-split flags from a command."""
    compact = command.replace("\\\n", " ").replace("\\", " ")
    compact = re.sub(r"\s+", " ", compact).strip()
    compact = _CROSS_VAL_RE.sub("", f" {compact}")
    compact = _CV_FOLD_RE.sub("", compact)
    compact = _VAL_SPLIT_RE.sub("", compact)
    compact = _ENSEMBLE_FOLDS_RE.sub("", compact)
    compact = _ENSEMBLE_METHOD_RE.sub("", compact)
    compact = _FOLD_DIRS_RE.sub("", compact)
    return re.sub(r"\s+", " ", compact).strip()


def _build_launch_variants(base_command: str, fold_count: int = DEFAULT_FOLD_COUNT) -> list[dict]:
    """Build no-val, fold, and ensemble variants for a base command."""
    base = _strip_cv_flags(base_command)
    variants = [{"id": "no_val", "label": "No Val", "command": base}]
    for fold in range(1, fold_count + 1):
        variants.append(
            {
                "id": f"fold_{fold}",
                "label": f"Fold {fold}",
                "command": f"{base} --cross_val {fold_count} --cv_fold {fold} --val_split kfold",
            }
        )
    variants.append(
        {
            "id": "ensemble",
            "label": "Ensemble",
            "command": f"{base} --mode infer --ensemble_folds",
        }
    )
    return variants


def _normalize_config_schema(config_data: dict) -> dict:
    """Ensure config schema includes launch variants, cv settings, and fold_state."""
    normalized = dict(config_data)
    cv = dict(normalized.get("cv", {}))
    fold_count = cv.get("fold_count", DEFAULT_FOLD_COUNT)
    if not isinstance(fold_count, int) or fold_count < 1:
        fold_count = DEFAULT_FOLD_COUNT
    cv["enabled"] = True
    cv["fold_count"] = fold_count
    normalized["cv"] = cv

    if normalized.get("command"):
        normalized["command"] = _strip_cv_flags(normalized["command"])
    normalized["launch_variants"] = _build_launch_variants(normalized["command"], fold_count)

    # Ensure fold_state dict exists with entries for each variant
    fold_state = dict(normalized.get("fold_state", {}))
    for variant in normalized["launch_variants"]:
        if variant["id"] not in fold_state:
            fold_state[variant["id"]] = {}
    # Migrate legacy top-level state into no_val fold
    no_val = fold_state.setdefault("no_val", {})
    if not no_val.get("status") and normalized.get("status"):
        no_val["status"] = normalized["status"]
    if not no_val.get("run_dir") and normalized.get("run_dir"):
        no_val["run_dir"] = normalized["run_dir"]
    if not no_val.get("original_run_dir") and normalized.get("original_run_dir"):
        no_val["original_run_dir"] = normalized["original_run_dir"]
    if not no_val.get("wandb_run_id") and normalized.get("wandb_run_id"):
        no_val["wandb_run_id"] = normalized["wandb_run_id"]
    normalized["fold_state"] = fold_state

    return normalized


_reset_stale_running()


def _get_checkpoint_epoch(config_data: dict, variant_id: str | None = None) -> int:
    """Return the highest completed epoch from a variant's run_dir, or 0."""
    run_dir = None
    if variant_id:
        run_dir = config_data.get("fold_state", {}).get(variant_id, {}).get("run_dir")
    # Only fall back to top-level run_dir for no_val (legacy configs)
    if not run_dir and (not variant_id or variant_id == "no_val"):
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
    path = get_config_path(config.name)
    path.parent.mkdir(parents=True, exist_ok=True)

    command = _strip_cv_flags(config.command)
    cv_payload = dict(config.cv or {})
    fold_count = cv_payload.get("fold_count", DEFAULT_FOLD_COUNT)
    if not isinstance(fold_count, int) or fold_count < 1:
        fold_count = DEFAULT_FOLD_COUNT

    config_data = {
        "name": config.name,
        "command": command,
        "params": config.params,
        "cv": {
            "enabled": True,
            "fold_count": fold_count,
        },
        "launch_variants": _build_launch_variants(command, fold_count),
    }
    config_data = _normalize_config_schema(config_data)

    with _config_lock:
        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)

    return {"success": True, "name": config.name}


@router.get("/api/configs/list")
async def list_configs():
    """List all saved configs."""
    configs = []

    for config_file in sorted(CONFIGS_DIR.glob("*.json")):
        with _config_lock:
            with open(config_file) as f:
                config_data = json.load(f)
        # Normalize for the response only — never write back during reads
        normalized = _normalize_config_schema(config_data)
        normalized["checkpoint_epoch"] = _get_checkpoint_epoch(normalized, "no_val")
        fold_ckpt = {}
        for variant in normalized.get("launch_variants", []):
            fold_ckpt[variant["id"]] = _get_checkpoint_epoch(normalized, variant["id"])
        normalized["fold_checkpoint_epochs"] = fold_ckpt
        configs.append(normalized)

    return configs


@router.get("/api/configs/get/{config_name}")
async def get_config(config_name: str):
    """Get a specific config."""
    path = get_config_path(config_name)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Config not found")

    with _config_lock:
        with open(path) as f:
            config_data = json.load(f)
    # Normalize for the response only — never write back during reads
    return _normalize_config_schema(config_data)


@router.patch("/api/configs/status/{config_name}")
async def update_config_status(config_name: str, status: str):
    """Update the status field of a config (idle, running, done)."""
    path = get_config_path(config_name)
    with _config_lock:
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {config_name}")
        with open(path) as f:
            config_data = json.load(f)
        config_data["status"] = status
        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)


def set_config_status(config_name: str, status: str):
    """Set config status synchronously (for use from drain thread)."""
    set_config_field(config_name, "status", status)


def set_config_field(config_name: str, key: str, value):
    """Set an arbitrary field on a config JSON file."""
    path = get_config_path(config_name)
    with _config_lock:
        if not path.exists():
            return
        with open(path) as f:
            config_data = json.load(f)
        config_data[key] = value
        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)


def set_fold_field(config_name: str, variant_id: str, key: str, value):
    """Set a field on a specific fold's state within a config."""
    path = get_config_path(config_name)
    with _config_lock:
        if not path.exists():
            return
        with open(path) as f:
            config_data = json.load(f)
        fold_state = config_data.setdefault("fold_state", {})
        fold_state.setdefault(variant_id, {})[key] = value
        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)


def set_fold_status(config_name: str, variant_id: str, status: str):
    """Set a fold's status synchronously."""
    set_fold_field(config_name, variant_id, "status", status)


def get_fold_state(config_name: str, variant_id: str) -> dict:
    """Get a fold's state dict from the config."""
    path = get_config_path(config_name)
    with _config_lock:
        if not path.exists():
            return {}
        with open(path) as f:
            config_data = json.load(f)
    return config_data.get("fold_state", {}).get(variant_id, {})


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
    for config_file in CONFIGS_DIR.glob("*.json"):
        with open(config_file) as f:
            data = json.load(f)
        name = data["name"]
        local_names.add(name)
        local_configs[name] = data

    deleted = []
    updated = []
    unchanged = []

    try:
        runs = api.runs("AutoMONAI")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list W&B runs: {e}")

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

    path.unlink()
    return {"success": True, "name": config_name}
