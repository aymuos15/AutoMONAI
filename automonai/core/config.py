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
    "basicunet": {
        "name": "BasicUNet",
        "description": "Simplified U-Net",
    },
    "basicunetplusplus": {
        "name": "BasicUNet++",
        "description": "UNet++ with nested skip connections",
    },
    "dynunet": {
        "name": "DynUNet",
        "description": "Dynamic U-Net (nnU-Net style)",
    },
    "vnet": {
        "name": "VNet",
        "description": "Volumetric segmentation network",
    },
    "highresnet": {
        "name": "HighResNet",
        "description": "High-resolution network for brain parcellation",
    },
    "unetr": {
        "name": "UNETR",
        "description": "UNet with Vision Transformers",
    },
    "segresnetvae": {
        "name": "SegResNetVAE",
        "description": "SegResNet with variational autoencoder",
    },
    "segresnetds": {
        "name": "SegResNetDS",
        "description": "SegResNet with deep supervision",
    },
    "segresnetds2": {
        "name": "SegResNetDS2",
        "description": "SegResNet deep supervision variant 2",
    },
    "flexibleunet": {
        "name": "FlexibleUNet",
        "description": "Configurable backbone U-Net (EfficientNet)",
    },
    "dints": {
        "name": "DiNTS",
        "description": "Differentiable Neural Architecture Search",
    },
    "mednext_s": {
        "name": "MedNeXt-S",
        "description": "ConvNeXt-style medical net (Small)",
    },
    "mednext_m": {
        "name": "MedNeXt-M",
        "description": "ConvNeXt-style medical net (Medium)",
    },
    "mednext_b": {
        "name": "MedNeXt-B",
        "description": "ConvNeXt-style medical net (Base)",
    },
    "mednext_l": {
        "name": "MedNeXt-L",
        "description": "ConvNeXt-style medical net (Large)",
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
    "inferer": "simple",
}

METRICS_AVAILABLE = [
    "dice",
    "iou",
    "hausdorff",
    "surface_distance",
    "surface_dice",
    "generalized_dice",
    "confusion_matrix",
    "rocauc",
    "fbeta",
    "panoptic_quality",
    "calibration",
]

LOSSES_AVAILABLE = [
    "dice",
    "cross_entropy",
    "focal",
    "dice_ce",
    "dice_focal",
    "generalized_dice",
    "generalized_wasserstein_dice",
    "generalized_dice_focal",
    "tversky",
    "hausdorff_dt",
    "log_hausdorff_dt",
    "soft_cl_dice",
    "soft_dice_cl_dice",
    "masked_dice",
    "nacl",
    "asymmetric_unified_focal",
    "ssim",
]

PREPROC_NORM_AVAILABLE = ["minmax", "zscore"]

PREPROC_CROP_AVAILABLE = ["center", "random"]

OPTIMIZERS_AVAILABLE = ["adam", "adamw", "sgd", "novograd", "rmsprop"]

SCHEDULERS_AVAILABLE = [
    "none",
    "cosine",
    "step",
    "plateau",
    "warmup_cosine",
    "cosine_warm_restarts",
    "polynomial",
]

INFERERS_AVAILABLE = ["simple", "sliding_window", "patch", "saliency", "slice"]

VAL_SPLIT_MODES = ["none", "holdout", "kfold", "custom"]
BEST_METRIC_CHOICES = ["val_loss", "val_dice", "val_iou", "train_loss"]
ENSEMBLE_METHODS = ["mean", "vote"]

AUGMENTATION_TRANSFORMS = [
    "rotate",
    "flip",
    "rand_affine",
    "rand_elastic_2d",
    "rand_elastic_3d",
    "rand_crop_pos_neg",
    "crop_foreground",
    "rand_rotate90",
    "rand_spatial_crop_samples",
    "spatial_pad",
    "border_pad",
    "divisible_pad",
    "grid_patch",
    "rand_gibbs_noise",
    "rand_kspace_spike_noise",
    "rand_bias_field",
    "rand_coarse_dropout",
    "rand_coarse_shuffle",
    "rand_histogram_shift",
    "rand_shift_intensity",
    "rand_scale_intensity",
    "rand_gaussian_smooth",
    "rand_gaussian_sharpen",
    "mask_intensity",
    "clip_intensity_percentiles",
    "scale_intensity_range",
    "threshold_intensity",
]

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
    "LMDBDataset": {
        "name": "LMDBDataset",
        "description": "LMDB-backed persistent cache for fast random access",
    },
    "CacheNTransDataset": {
        "name": "CacheNTransDataset",
        "description": "Cache first N transforms for partial caching",
    },
    "ArrayDataset": {
        "name": "ArrayDataset",
        "description": "Simple dataset from image/label arrays",
    },
    "ZipDataset": {
        "name": "ZipDataset",
        "description": "Combine multiple datasets together",
    },
    "GridPatchDataset": {
        "name": "GridPatchDataset",
        "description": "Grid-based patch extraction dataset",
    },
    "PatchDataset": {
        "name": "PatchDataset",
        "description": "Random patch extraction dataset",
    },
    "DecathlonDataset": {
        "name": "DecathlonDataset",
        "description": "Medical Segmentation Decathlon dataset",
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
