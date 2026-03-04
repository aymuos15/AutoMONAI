# Configs

A config is a saved training command with status tracking. The UI creates them automatically, but you can also create and manage them programmatically.

## Config JSON

Configs are stored as JSON files in `UI/configs/`. Each file looks like:

```json
{
  "name": "model_segresnet_epochs_8",
  "command": "python3 -m src.run --dataset Dataset001_Cellpose --model segresnet --epochs 8",
  "params": {
    "model": "segresnet",
    "dataset": "Dataset001_Cellpose",
    "epochs": "8"
  },
  "status": "done",
  "run_dir": "results/Dataset001_Cellpose/segresnet/20260304_165949",
  "original_run_dir": "results/Dataset001_Cellpose/segresnet/20260304_165949"
}
```

### Fields

| Field | Set by | Description |
|---|---|---|
| `name` | User/auto | Unique identifier. Used as filename and W&B run ID. |
| `command` | User | Full CLI command to execute. Must start with `python3 -m src.run`. |
| `params` | User | Extracted subset of command params (model, dataset, epochs). Used for W&B sync. |
| `status` | System | One of `idle`, `running`, `done`. Reset to `idle` on server restart. |
| `run_dir` | System | Path to the results directory created during training. Set by the launch drain thread. |
| `original_run_dir` | System | Preserved across re-launches. Used by auto-resume to find checkpoints. |
| `checkpoint_epoch` | Computed | Not stored — computed at list time by scanning `run_dir/checkpoints/epoch_*.pt`. |

## Creating a config via API

```bash
curl -X POST http://localhost:8888/api/configs/save \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_experiment",
    "command": "python3 -m src.run --dataset Dataset001_Cellpose --model unet --epochs 10 --lr 0.001",
    "params": {"model": "unet", "dataset": "Dataset001_Cellpose", "epochs": "10"}
  }'
```

## Creating a config via Python

```python
import json
from pathlib import Path

config = {
    "name": "my_experiment",
    "command": "python3 -m src.run --dataset Dataset001_Cellpose --model unet --epochs 10 --lr 0.001",
    "params": {"model": "unet", "dataset": "Dataset001_Cellpose", "epochs": "10"},
}

path = Path("UI/configs/my_experiment.json")
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(config, indent=2))
```

The config will appear in the Configs tab on next page load.

## CLI flags reference

Run `python3 -m src.run --help` for full details.

```
--mode                train | infer (default: train)
--dataset             Dataset name (nnUNet format, e.g. Dataset001_Cellpose)
--model               unet | attention_unet | segresnet | swinunetr

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
--loss                dice | cross_entropy | focal (default: dice)
--metrics             dice iou (space-separated, default: dice iou)
--optimizer           adam | adamw | sgd (default: adam)
--scheduler           none | cosine | step | plateau (default: none)
--mixed_precision     no | fp16 | bf16 (default: no)
--early_stopping      true | false (default: false)
--patience            Epochs without improvement before stopping (default: 5)

Preprocessing / Augmentation:
--norm                minmax zscore none (space-separated)
--crop                center random none (space-separated)
--augment             true | false (default: false)
--aug_prob            Probability per augmentation transform (default: 0.5)

Dataset classes:
--train_dataset_class       Dataset | CacheDataset | PersistentDataset | SmartCacheDataset
--inference_dataset_class   Dataset | CacheDataset | PersistentDataset | SmartCacheDataset
--cache_rate                Cache rate for CacheDataset/SmartCacheDataset (default: 1.0)
--smart_replace_rate        Replace rate for SmartCacheDataset
--cache_dir                 Cache directory for PersistentDataset

Resume / Inference:
--resume              Path to run directory (e.g. results/dataset/model/timestamp)
--checkpoint          Checkpoint file to load (default: best_model.pt)
--save_predictions    Save prediction images to output_dir
--output_dir          Output directory for predictions

W&B:
--run_id              Stable run ID (reuses same W&B run on re-launch)

Info:
--show_config         Print all available datasets and models
--list_datasets       List available datasets as JSON
```

## Config lifecycle

```
[Create] → idle → [Launch] → running → [Complete] → done → [Infer] → running → done
                                      ↘ [Stop] → idle (checkpoint preserved)
```

- **idle**: Ready to launch. Shows Launch button.
- **running**: Process active. Shows Stop button + progress bar.
- **done**: Training completed. Shows Infer button. Auto-resume available.
