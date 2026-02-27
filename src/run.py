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
from .results import RunLogger
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

    # Initialize run logger
    logger = RunLogger(dataset_name, args.model)

    # Save configuration
    config = {
        "dataset": dataset_name,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "image_size": args.img_size,
        "num_workers": args.num_workers,
        "loss": args.loss,
        "metrics": args.metrics,
        "device": args.device,
        "spatial_dims": spatial_dims,
        "train_dataset_class": args.train_dataset_class,
        "inference_dataset_class": args.inference_dataset_class,
        "normalization": args.norm,
        "crop": args.crop,
        "augmentation_enabled": args.augment,
        "optimizer": args.optimizer,
        "mixed_precision": args.mixed_precision,
        "scheduler": args.scheduler,
        "early_stopping_enabled": args.early_stopping,
        "patience": args.patience,
    }
    logger.save_config(config)

    # Initialize Fabric for device and distributed training management
    precision = args.mixed_precision if args.mixed_precision != "no" else None
    fabric = Fabric(accelerator="auto" if not args.device else args.device, precision=precision)
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

    # Create optimizer based on user selection
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup model, optimizer, and dataloaders with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Setup learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, args.patience // 2))
    else:
        scheduler = None

    # Early stopping state
    no_improve_count = 0

    print(f"\nTraining for {args.epochs} epoch(s)...")
    best_loss = float("inf")
    for epoch in range(args.epochs):
        result = train_one_epoch(fabric, model, train_loader, loss_fn, optimizer, args.metrics)
        loss = result["loss"]

        # Log epoch metrics
        epoch_metrics = {"loss": loss}
        if "dice" in result:
            epoch_metrics["dice"] = result["dice"]
        if "iou" in result:
            epoch_metrics["iou"] = result["iou"]

        logger.log_epoch(epoch, epoch_metrics)

        # Check if best model
        is_best = loss < best_loss
        if is_best:
            best_loss = loss

        # Save checkpoint
        logger.save_checkpoint(model, epoch, optimizer, is_best=is_best)

        # Step learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        # Early stopping check
        if args.early_stopping:
            if not is_best:
                no_improve_count += 1
                if no_improve_count >= args.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {args.patience} epochs).")
                    break
            else:
                no_improve_count = 0

        log_msg = f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f}"
        if "dice" in result:
            log_msg += f" - Dice: {result['dice']:.4f}"
        if "iou" in result:
            log_msg += f" - IoU: {result['iou']:.4f}"
        print(log_msg)

    # Save final summary
    summary = {
        "total_epochs": args.epochs,
        "best_loss": best_loss,
        "final_metrics": epoch_metrics,
        "dataset": dataset_name,
        "model": args.model,
    }
    logger.save_final_summary(summary)

    if args.save_predictions:
        save_dir = Path(args.output_dir) / dataset_name / args.model
        print(f"\nRunning inference on {len(test_files)} test samples...")
        print(f"Saving predictions to: {save_dir}")
        infer(fabric, model, test_loader, str(save_dir), spatial_dims)

    print(f"\nTraining results saved to: {logger.run_dir}")
    logger.close()
    print("Done!")


if __name__ == "__main__":
    main()
