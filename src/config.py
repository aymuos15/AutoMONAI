DATASET_ROOT = "/home/localssk23/CAI4Soumya/SegData/nnUNet_raw"

OUTPUT_DEFAULTS = {
    "default_output_dir": "/home/localssk23/MonaiUI/predictions",
}

MODELS = {
    "unet": {
        "name": "UNet",
        "description": "Standard U-Net architecture",
    },
    "attention_unet": {
        "name": "Attention UNet",
        "description": "U-Net with attention gates",
    },
    "segresnet": {
        "name": "SegResNet",
        "description": "Segmentation ResNet",
    },
    "swinunetr": {
        "name": "Swin UNETR",
        "description": "Swin Transformer-based UNETR",
    },
}

TRAINING_DEFAULTS = {
    "epochs": 1,
    "batch_size": 4,
    "lr": 1e-4,
    "img_size": 128,
    "num_workers": 0,
    "spatial_dims": 2,
    "val_interval": 1,
    "metrics": ["dice", "iou"],
    "loss": "dice",
    "optimizer": "adam",
    "mixed_precision": "no",
    "early_stopping": False,
    "patience": 5,
    "scheduler": "none",
}

METRICS_AVAILABLE = ["dice", "iou"]

LOSSES_AVAILABLE = ["dice", "cross_entropy", "focal"]

PREPROC_NORM_AVAILABLE = ["minmax", "zscore"]

PREPROC_CROP_AVAILABLE = ["center", "random"]

OPTIMIZERS_AVAILABLE = ["adam", "adamw", "sgd"]

SCHEDULERS_AVAILABLE = ["none", "cosine", "step", "plateau"]

DATASET_CLASSES = {
    "Dataset": {
        "name": "Dataset",
        "description": "Basic in-memory dataset",
    },
    "CacheDataset": {
        "name": "CacheDataset",
        "description": "Caches entire dataset in memory for fast access",
    },
    "PersistentDataset": {
        "name": "PersistentDataset",
        "description": "Persistent cache on disk with hash-based validity checking",
    },
    "SmartCacheDataset": {
        "name": "SmartCacheDataset",
        "description": "Intelligent caching with automatic replacement",
    },
}


def get_datasets():
    """Dynamically discover datasets from DATASET_ROOT."""
    import json
    from pathlib import Path

    datasets = {}
    root = Path(DATASET_ROOT)

    if not root.exists():
        return {}

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith("Dataset"):
            continue

        name = dataset_dir.name
        json_path = dataset_dir / "dataset.json"

        if not json_path.exists():
            continue

        try:
            with open(json_path) as f:
                info = json.load(f)
            datasets[name] = {
                "name": name,
                "description": info.get("description", info.get("name", "")),
                "channels": list(info.get("channel_names", {}).values()),
                "labels": info.get("labels", {}),
            }
        except Exception:
            continue

    return datasets


DATASETS = get_datasets()
