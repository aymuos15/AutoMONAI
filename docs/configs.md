# Configs

A config is a saved training command with status tracking. The UI creates them automatically, but you can also create and manage them programmatically.

## Config JSON

Configs are stored as JSON files in `automonai/backend/configs/`. Each file looks like:

```json
{
  "name": "model_segresnet_epochs_8",
  "command": "python3 -m automonai.core.run --dataset Dataset001_Cellpose --model segresnet --epochs 8",
  "params": {
    "model": "segresnet",
    "dataset": "Dataset001_Cellpose",
    "epochs": "8"
  },
  "cv": {
    "enabled": true,
    "fold_count": 5
  },
  "launch_variants": [
    { "id": "no_val", "label": "No Val", "command": "..." },
    { "id": "fold_1", "label": "Fold 1", "command": "... --cross_val 5 --cv_fold 1" }
  ],
  "fold_state": {
    "no_val": {
      "status": "done",
      "run_dir": "results/Dataset001_Cellpose/segresnet/20260304_165949",
      "original_run_dir": "results/Dataset001_Cellpose/segresnet/20260304_165949",
      "wandb_run_id": "abc123"
    },
    "fold_1": {},
    "fold_2": {}
  }
}
```

### Fields

| Field | Set by | Description |
|---|---|---|
| `name` | User/auto | Unique identifier. Used as filename and W&B run display name. |
| `command` | User | Base CLI command (CV flags stripped). Must start with `python3 -m automonai.core.run`. |
| `params` | User | Extracted subset of command params (model, dataset, epochs). Used for W&B sync. |
| `cv` | System | Cross-validation settings: `enabled` (bool), `fold_count` (int, default 5). |
| `launch_variants` | System | Auto-generated list of launchable variants: `no_val` (base command) plus one per fold with `--cross_val K --cv_fold N` appended. |
| `fold_state` | System | Per-variant state dict keyed by variant ID (`no_val`, `fold_1`, etc.). Each entry contains: |
| `fold_state.*.status` | System | One of `idle`, `running`, `done`, `inferred`. Reset to `idle` on server restart. |
| `fold_state.*.run_dir` | System | Path to the results directory created during training. Set by the launch drain thread. |
| `fold_state.*.original_run_dir` | System | Preserved across re-launches. Used by auto-resume to find checkpoints. |
| `fold_state.*.wandb_run_id` | System | W&B auto-generated run ID. Each fold gets its own W&B run. Reused for inference. |
| `checkpoint_epoch` | Computed | Not stored — computed at list time by scanning `fold_state.*.run_dir/checkpoints/epoch_*.pt`. |
| `fold_checkpoint_epochs` | Computed | Not stored — per-variant checkpoint epochs returned by list endpoint. |

## Creating a config via API

```bash
curl -X POST http://localhost:8888/api/configs/save \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_experiment",
    "command": "python3 -m automonai.core.run --dataset Dataset001_Cellpose --model unet --epochs 10 --lr 0.001",
    "params": {"model": "unet", "dataset": "Dataset001_Cellpose", "epochs": "10"},
    "cv": {"enabled": true, "fold_count": 5}
  }'
```

The server auto-generates `launch_variants` and `fold_state` on save.

## Creating a config via Python

```python
import json
from pathlib import Path

config = {
    "name": "my_experiment",
    "command": "python3 -m automonai.core.run --dataset Dataset001_Cellpose --model unet --epochs 10 --lr 0.001",
    "params": {"model": "unet", "dataset": "Dataset001_Cellpose", "epochs": "10"},
}

path = Path("automonai/backend/configs/my_experiment.json")
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(config, indent=2))
```

The config will appear in the Configs tab on next page load.

## CLI flags reference

Run `python3 -m automonai.core.run --help` for full details.

```
--mode                train | infer (default: train)
--dataset             Dataset name (nnUNet format, e.g. Dataset001_Cellpose)
--model               unet | attention_unet | segresnet | swinunetr | basicunet | basicunetplusplus | dynunet | vnet | highresnet | unetr | segresnetvae | segresnetds | segresnetds2 | flexibleunet | dints | mednext_s | mednext_m | mednext_b | mednext_l

Training:
--epochs              Number of training epochs (default: 1)
--batch_size          Batch size (default: 4)
--lr                  Learning rate (default: 0.0001)
--img_size            Resize images to NxN (default: 128)
--num_workers         Data loading workers (default: 0)
--val_interval        Validation interval (default: 1)
--spatial_dims        2 | 3 (default: 2)
--device              cuda | cpu (default: auto)

Loss / Metrics / Optimizer:
--loss                dice | cross_entropy | focal | dice_ce | dice_focal | generalized_dice |
                      generalized_wasserstein_dice | generalized_dice_focal | tversky |
                      hausdorff_dt | log_hausdorff_dt | soft_cl_dice | soft_dice_cl_dice |
                      masked_dice | nacl | asymmetric_unified_focal | ssim (default: dice)
--deep_supervision    true | false — wraps loss with DeepSupervisionLoss (default: false)
--metrics             dice iou hausdorff_distance surface_distance surface_dice
                      generalized_dice confusion_matrix fbeta panoptic_quality
                      (space-separated, default: dice iou)
--optimizer           adam | adamw | sgd | novograd | rmsprop (default: adam)
--scheduler           none | cosine | step | plateau | warmup_cosine |
                      cosine_warm_restarts | polynomial (default: none)
--mixed_precision     no | fp16 | bf16 (default: no)
--early_stopping      true | false (default: false)
--patience            Epochs without improvement before stopping (default: 5)
--inferer             simple | sliding_window | patch | saliency | slice (default: simple)

Preprocessing / Augmentation:
--norm                minmax zscore none (space-separated)
--crop                center random none (space-separated)
--augment             true | false (default: false)
--aug_prob            Probability per augmentation transform (default: 0.5)
--extra_transforms    Space-separated list of additional MONAI transforms
                      (e.g. rand_affine rand_gibbs_noise rand_elastic_2d)

Dataset classes:
--train_dataset_class       Dataset | CacheDataset | PersistentDataset | SmartCacheDataset |
                            LMDBDataset | CacheNTransDataset | ArrayDataset | ZipDataset |
                            GridPatchDataset | PatchDataset | DecathlonDataset
--inference_dataset_class   (same options as train)
--cache_rate                Cache rate for training CacheDataset/SmartCacheDataset (default: 1.0)
--smart_replace_rate        Replace rate for SmartCacheDataset
--cache_dir                 Cache directory for training PersistentDataset/LMDBDataset/CacheNTransDataset
--inference_cache_rate      Cache rate for inference (defaults to --cache_rate)
--inference_cache_dir       Cache directory for inference (defaults to --cache_dir)

Cross-validation:
--cross_val           Number of folds for K-fold CV (e.g. 5)
--cv_fold             Run only fold N (1-based, used with --cross_val)

Resume / Inference:
--resume              Path to run directory (e.g. results/dataset/model/timestamp)
--checkpoint          Checkpoint file to load (default: best_model.pt)
--save_predictions    Save prediction images to output_dir
--output_dir          Output directory for predictions

W&B:
--run_id              Config name passed by the UI (used as W&B display name)
--wandb_run_id        W&B run ID to resume (set automatically by the UI)

Info:
--show_config         Print all available datasets and models
--list_datasets       List available datasets as JSON
```

## Config lifecycle

Each fold variant has an independent lifecycle tracked in `fold_state`:

```
[Create] → idle → [Launch] → running → [Complete] → done → [Infer] → running → inferred
                                      ↘ [Stop] → idle (checkpoint preserved)
                                        done → [Re-launch] → running (starts fresh)
                                        inferred → [Re-launch] → running (starts fresh)
```

- **idle**: Ready to launch. Shows Launch button. Card border: red.
- **running**: Process active. Shows Stop button + progress bar. Card border: yellow.
- **done**: Training completed. Shows Launch + Infer buttons. Card border: yellow.
- **inferred**: Inference completed. Shows Launch button. Card border: green.

Re-launching a completed config starts training from scratch (no auto-resume). Auto-resume only applies when the previous run was interrupted before reaching the target epoch count.

The dropdown on each config card selects the active variant (`No Val`, `Fold 1`, ..., `Fold K`). Each variant tracks its own status, progress, run directory, and W&B run independently. Multiple folds can run concurrently — file access is serialized via a threading lock to prevent race conditions.
