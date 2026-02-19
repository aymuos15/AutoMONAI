#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from monai.data import DataLoader
from lightning.fabric import Fabric

from .config import DATASETS
from .cli import get_parser, print_config
from .dataset import (
    get_train_files,
    get_test_files,
    create_train_dataset,
    create_inference_dataset,
    TrainDataset,
    TestDataset,
    DictTransform,
)
from .models import get_model
from .inference import infer
from .train import train_one_epoch, get_loss
from .transforms import get_transforms


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.show_config:
        print_config()
        sys.exit(0)

    if args.list_datasets:
        import json

        datasets = {}
        for key, val in DATASETS.items():
            datasets[key] = {
                "name": val["name"],
                "channels": val["channels"],
                "labels": val["labels"],
            }
        print(json.dumps(datasets))
        sys.exit(0)

    dataset_name = args.dataset
    dataset_info = DATASETS[dataset_name]

    print(f"\nSelected dataset: {dataset_name}")
    print(f"  Name: {dataset_info['name']}")
    print(f"  Description: {dataset_info['description']}")
    print(f"  Channels: {dataset_info['channels']}")
    print(f"  Labels: {dataset_info['labels']}")

    in_channels = len(dataset_info["channels"])
    out_channels = len(dataset_info["labels"])

    print(f"\nSelected model: {args.model}")
    print(f"  In channels: {in_channels}, Out channels: {out_channels}")

    train_files, is_3d = get_train_files(dataset_name)
    test_files, _ = get_test_files(dataset_name)

    spatial_dims = 3 if is_3d else 2
    print(f"  Detected: {'3D' if is_3d else '2D'} dataset")

    print(f"Training files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    if not train_files:
        print("No training files found!")
        sys.exit(1)

    print(f"\nDataset class (train): {args.train_dataset_class}")
    print(f"Dataset class (inference): {args.inference_dataset_class}")
    if args.cache_rate < 1.0:
        print(f"Cache rate: {args.cache_rate}")
    if args.smart_replace_rate:
        print(f"Smart cache replace rate: {args.smart_replace_rate}")
    if args.cache_dir:
        print(f"Cache dir: {args.cache_dir}")

    # Initialize Fabric for device and distributed training management
    fabric = Fabric(accelerator="auto" if not args.device else args.device)
    fabric.launch()

    print(f"Using device: {fabric.device}")

    train_transforms = get_transforms(
        args.img_size, spatial_dims, norm=args.norm, crop=args.crop, is_train=True
    )
    label_transforms = get_transforms(
        args.img_size, spatial_dims, norm=args.norm, crop=args.crop, is_train=True
    )
    test_transforms = get_transforms(
        args.img_size, spatial_dims, norm=args.norm, crop=args.crop, is_train=False
    )

    if args.train_dataset_class == "Dataset":
        train_ds = TrainDataset(
            train_files, transform=train_transforms, label_transform=label_transforms
        )
    else:
        dict_transform = DictTransform(train_transforms, label_transforms)
        train_ds = create_train_dataset(
            args.train_dataset_class,
            train_files,
            dict_transform,
            label_transforms,
            cache_rate=args.cache_rate,
            replace_rate=args.smart_replace_rate,
            cache_dir=args.cache_dir,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    if args.inference_dataset_class == "Dataset":
        test_ds = TestDataset(test_files, transform=test_transforms)
    else:
        test_ds = create_inference_dataset(
            args.inference_dataset_class,
            test_files,
            test_transforms,
            cache_rate=args.cache_rate,
            cache_dir=args.cache_dir,
        )

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.model, in_channels, out_channels, spatial_dims, args.img_size)
    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup model, optimizer, and dataloaders with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    print(f"\nTraining for {args.epochs} epoch(s)...")
    for epoch in range(args.epochs):
        result = train_one_epoch(fabric, model, train_loader, loss_fn, optimizer, args.metrics)
        loss = result["loss"]
        log_msg = f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f}"
        if "dice" in result:
            log_msg += f" - Dice: {result['dice']:.4f}"
        if "iou" in result:
            log_msg += f" - IoU: {result['iou']:.4f}"
        print(log_msg)

    save_dir = Path(args.output_dir) / dataset_name / args.model
    print(f"\nRunning inference on {len(test_files)} test samples...")
    print(f"Saving predictions to: {save_dir}")
    infer(fabric, model, test_loader, str(save_dir), spatial_dims)

    print("\nDone!")


if __name__ == "__main__":
    main()
