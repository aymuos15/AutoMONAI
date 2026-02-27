import argparse

from .config import (
    DATASETS,
    MODELS,
    TRAINING_DEFAULTS,
    OUTPUT_DEFAULTS,
    METRICS_AVAILABLE,
    LOSSES_AVAILABLE,
    PREPROC_NORM_AVAILABLE,
    PREPROC_CROP_AVAILABLE,
)
from .dataset import list_datasets


def print_config():
    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS:")
    print("=" * 60)
    for key, val in DATASETS.items():
        print(f"\n{key} ({val['name']})")
        print(f"  Description: {val['description']}")
        print(f"  Channels: {', '.join(val['channels'])}")
        print(f"  Labels: {val['labels']}")

    print("\n" + "=" * 60)
    print("AVAILABLE MODELS:")
    print("=" * 60)
    for key, val in MODELS.items():
        print(f"\n{key} ({val['name']})")
        print(f"  Description: {val['description']}")

    print("\n" + "=" * 60)
    print("TRAINING DEFAULTS:")
    print("=" * 60)
    for key, val in TRAINING_DEFAULTS.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("OUTPUT DEFAULTS:")
    print("=" * 60)
    print(f"  default_output_dir: {OUTPUT_DEFAULTS['default_output_dir']}")
    print("=" * 60 + "\n")


def get_parser():
    parser = argparse.ArgumentParser(
        description="MONAI Training and Inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list_datasets(),
        help=f"Dataset to use. Available: {', '.join(list_datasets())}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=list(MODELS.keys()),
        help=f"Model to use. Available: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_DEFAULTS["epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRAINING_DEFAULTS["batch_size"],
        help="Batch size",
    )
    parser.add_argument("--lr", type=float, default=TRAINING_DEFAULTS["lr"], help="Learning rate")
    parser.add_argument(
        "--img_size",
        type=int,
        default=TRAINING_DEFAULTS["img_size"],
        help="Image size (will resize to img_size x img_size)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=TRAINING_DEFAULTS["num_workers"],
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=TRAINING_DEFAULTS["val_interval"],
        help="Validation interval",
    )
    parser.add_argument(
        "--spatial_dims",
        type=int,
        default=TRAINING_DEFAULTS["spatial_dims"],
        choices=[2, 3],
        help="Spatial dimensions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DEFAULTS["default_output_dir"],
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--train_dataset_class",
        type=str,
        default="Dataset",
        choices=["Dataset", "CacheDataset", "PersistentDataset", "SmartCacheDataset"],
        help="Dataset class for training",
    )
    parser.add_argument(
        "--inference_dataset_class",
        type=str,
        default="Dataset",
        choices=["Dataset", "CacheDataset", "PersistentDataset", "SmartCacheDataset"],
        help="Dataset class for inference",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for PersistentDataset",
    )
    parser.add_argument(
        "--cache_rate",
        type=float,
        default=1.0,
        help="Cache rate for CacheDataset/SmartCacheDataset (0.0 to 1.0)",
    )
    parser.add_argument(
        "--smart_replace_rate",
        type=float,
        default=None,
        help="Replace rate for SmartCacheDataset (fraction of cache to replace per epoch)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        help="Show all available datasets and models",
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="List available datasets as JSON",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=TRAINING_DEFAULTS["metrics"],
        choices=METRICS_AVAILABLE,
        help=f"Metrics to compute during training/validation. Available: {', '.join(METRICS_AVAILABLE)}",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=TRAINING_DEFAULTS["loss"],
        choices=LOSSES_AVAILABLE,
        help=f"Loss function to use. Available: {', '.join(LOSSES_AVAILABLE)}",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Run inference and save predictions after training",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--aug_prob",
        type=float,
        default=0.5,
        help="Probability for each augmentation transform (0.0 to 1.0)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        nargs="+",
        default=[],
        choices=PREPROC_NORM_AVAILABLE,
        help=f"Normalization methods to apply. Available: {', '.join(PREPROC_NORM_AVAILABLE)}",
    )
    parser.add_argument(
        "--crop",
        type=str,
        nargs="+",
        default=[],
        choices=PREPROC_CROP_AVAILABLE,
        help=f"Crop methods to apply. Available: {', '.join(PREPROC_CROP_AVAILABLE)}",
    )

    return parser
