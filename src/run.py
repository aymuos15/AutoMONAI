#!/usr/bin/env python3

import atexit
import signal
import sys
from pathlib import Path

import torch
import wandb


def _cleanup_wandb(*_):
    """Ensure W&B flushes data on termination."""
    try:
        wandb.finish()  # type: ignore[unresolved-attribute]
    except Exception:
        pass


atexit.register(_cleanup_wandb)
signal.signal(signal.SIGTERM, lambda *_: (_cleanup_wandb(), sys.exit(0)))
from monai.data import DataLoader  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from .config import DATASETS  # noqa: E402
from .cli import get_parser, print_config  # noqa: E402
from .dataset import (  # noqa: E402
    get_train_files,
    get_test_files,
    get_test_files_with_labels,
    create_train_dataset,
    create_inference_dataset,
    TrainDataset,
    TestDataset,
    DictTransform,
)
from .models import get_model  # noqa: E402
from .inference import infer, infer_with_metrics  # noqa: E402
from .results import RunLogger  # noqa: E402
from .train import train_one_epoch, get_loss  # noqa: E402
from .transforms import get_transforms  # noqa: E402


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Convert string booleans to actual bools
    args.augment = args.augment == "true"
    args.early_stopping = args.early_stopping == "true"

    # Filter out "none" from list args
    args.norm = [v for v in args.norm if v != "none"]
    args.crop = [v for v in args.crop if v != "none"]

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

    if args.mode == "infer" and not args.resume:
        print("Error: --resume is required for inference mode")
        sys.exit(1)

    resume_from = args.resume
    resume_checkpoint = args.checkpoint
    loaded_config = None
    start_epoch = 0

    if resume_from:
        print(f"\n=== Resuming from: {resume_from} ===")
        print(f"Using checkpoint: {resume_checkpoint}")

        checkpoint_path = RunLogger.get_checkpoint_path(resume_from, resume_checkpoint)
        checkpoint_data = RunLogger.load_checkpoint(str(checkpoint_path))

        start_epoch = checkpoint_data["epoch"]
        print(f"Resuming from epoch: {start_epoch}")

        # Load config from previous run if available, otherwise use CLI args
        try:
            loaded_config = RunLogger.load_run_config(resume_from)
            dataset_name = loaded_config["dataset"]
        except FileNotFoundError:
            print("No config.json in run directory, using CLI args")
            loaded_config = None
            dataset_name = args.dataset
    else:
        dataset_name = args.dataset
    dataset_info = DATASETS[dataset_name]

    print(f"\nSelected dataset: {dataset_name}")
    print(f"  Name: {dataset_info['name']}")
    print(f"  Description: {dataset_info['description']}")
    print(f"  Channels: {dataset_info['channels']}")
    print(f"  Labels: {dataset_info['labels']}")

    in_channels = len(dataset_info["channels"])
    out_channels = len(dataset_info["labels"])

    if loaded_config:
        model_name = loaded_config["model"]
        train_dataset_class = loaded_config.get("train_dataset_class", "Dataset")
        inference_dataset_class = loaded_config.get("inference_dataset_class", "Dataset")
        cache_rate = loaded_config.get("cache_rate", 1.0)
        smart_replace_rate = loaded_config.get("smart_replace_rate")
        cache_dir = loaded_config.get("cache_dir")
        img_size = loaded_config.get("image_size", args.img_size)
    else:
        model_name = args.model
        train_dataset_class = args.train_dataset_class
        inference_dataset_class = args.inference_dataset_class
        cache_rate = args.cache_rate
        smart_replace_rate = args.smart_replace_rate
        cache_dir = args.cache_dir
        img_size = args.img_size

    print(f"\nSelected model: {model_name}")
    print(f"  In channels: {in_channels}, Out channels: {out_channels}")

    train_files, is_3d = get_train_files(dataset_name)
    test_files, _ = get_test_files(dataset_name)

    spatial_dims = 3 if is_3d else 2
    print(f"  Detected: {'3D' if is_3d else '2D'} dataset")

    print(f"Training files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    if args.mode == "train" and not train_files:
        print("No training files found!")
        sys.exit(1)

    print(f"\nDataset class (train): {train_dataset_class}")
    print(f"Dataset class (inference): {inference_dataset_class}")
    if resume_from:
        print(f"Cache rate: {cache_rate}")
    elif args.cache_rate < 1.0:
        print(f"Cache rate: {args.cache_rate}")
    if resume_from:
        print(f"Smart cache replace rate: {smart_replace_rate or 'N/A'}")
    elif args.smart_replace_rate:
        print(f"Smart cache replace rate: {args.smart_replace_rate}")
    if resume_from:
        print(f"Cache dir: {cache_dir or 'N/A'}")
    elif args.cache_dir:
        print(f"Cache dir: {args.cache_dir}")

    # Initialize run logger
    logger = RunLogger(dataset_name, model_name, resume_from=resume_from)

    # Build configuration
    if loaded_config:
        config = loaded_config.copy()
        config["epochs"] = args.epochs
        config["learning_rate"] = args.lr
        config["batch_size"] = args.batch_size
        config["num_workers"] = loaded_config.get("num_workers", args.num_workers)
        config["loss"] = loaded_config.get("loss", args.loss)
        config["metrics"] = loaded_config.get("metrics", args.metrics)
        config["device"] = loaded_config.get("device", args.device)
        config["image_size"] = img_size
        config["train_dataset_class"] = train_dataset_class
        config["inference_dataset_class"] = inference_dataset_class
        config["cache_rate"] = cache_rate
        config["smart_replace_rate"] = smart_replace_rate
        config["cache_dir"] = cache_dir
        config["normalization"] = loaded_config.get("normalization", [])
        config["crop"] = loaded_config.get("crop", [])
        config["augmentation_enabled"] = loaded_config.get("augmentation_enabled", False)
        config["optimizer"] = args.optimizer
        config["scheduler"] = args.scheduler
        config["mixed_precision"] = args.mixed_precision
        config["early_stopping_enabled"] = args.early_stopping
        config["patience"] = args.patience
        config["resume_from"] = resume_from
        config["resume_checkpoint"] = resume_checkpoint
        config["start_epoch"] = start_epoch
    else:
        config = {
            "dataset": dataset_name,
            "model": model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": args.img_size,
            "num_workers": args.num_workers,
            "loss": args.loss,
            "metrics": args.metrics,
            "device": args.device,
            "spatial_dims": spatial_dims,
            "train_dataset_class": train_dataset_class,
            "inference_dataset_class": inference_dataset_class,
            "normalization": args.norm,
            "crop": args.crop,
            "augmentation_enabled": args.augment,
            "optimizer": args.optimizer,
            "mixed_precision": args.mixed_precision,
            "scheduler": args.scheduler,
            "early_stopping_enabled": args.early_stopping,
            "patience": args.patience,
        }
    if resume_from:
        config["resume_from"] = resume_from
        config["resume_checkpoint"] = resume_checkpoint
        config["start_epoch"] = start_epoch

    # Save config to run directory for future resume
    logger.save_config(config)
    wandb_kwargs = {"project": "AutoMONAI", "config": config}
    if args.run_id:
        wandb_kwargs["id"] = args.run_id
        wandb_kwargs["resume"] = "allow"
        wandb_kwargs["name"] = args.run_id
    else:
        wandb_kwargs["name"] = f"{dataset_name}_{model_name}"
    wandb.init(**wandb_kwargs)  # type: ignore[unresolved-attribute]

    # Initialize Fabric for device and distributed training management
    _precision_map = {"fp16": "16-mixed", "bf16": "bf16-mixed"}
    precision = _precision_map.get(args.mixed_precision) if args.mixed_precision != "no" else None
    fabric = Fabric(accelerator="auto" if not args.device else args.device, precision=precision)  # type: ignore[invalid-argument-type]
    fabric.launch()

    print(f"Using device: {fabric.device}")

    train_transforms = get_transforms(
        img_size, spatial_dims, norm=args.norm, crop=args.crop, is_train=True
    )
    label_transforms = get_transforms(
        img_size, spatial_dims, norm=args.norm, crop=args.crop, is_train=True
    )
    test_transforms = get_transforms(
        img_size, spatial_dims, norm=args.norm, crop=args.crop, is_train=False
    )

    if train_dataset_class == "Dataset":
        train_ds = TrainDataset(
            train_files, transform=train_transforms, label_transform=label_transforms
        )
    else:
        dict_transform = DictTransform(train_transforms, label_transforms)
        train_ds = create_train_dataset(
            train_dataset_class,
            train_files,
            dict_transform,
            label_transforms,
            cache_rate=cache_rate,
            replace_rate=smart_replace_rate,
            cache_dir=cache_dir,
        )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]
    )

    if inference_dataset_class == "Dataset":
        test_ds = TestDataset(test_files, transform=test_transforms)
    else:
        test_ds = create_inference_dataset(
            inference_dataset_class,
            test_files,
            test_transforms,
            cache_rate=cache_rate,
            cache_dir=cache_dir,
        )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]
    )

    model = get_model(model_name, in_channels, out_channels, spatial_dims, config["image_size"])
    loss_fn = get_loss(config["loss"])

    # Create optimizer based on config
    if config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Setup model, optimizer, and dataloaders with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Load checkpoint if resuming
    if resume_from:
        checkpoint_path = RunLogger.get_checkpoint_path(resume_from, resume_checkpoint)
        checkpoint_data = RunLogger.load_checkpoint(str(checkpoint_path))
        model.load_state_dict(checkpoint_data["model_state"])
        if checkpoint_data["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")

    # Inference mode: evaluate test set with metrics, log to W&B, exit
    if args.mode == "infer":
        test_files_labeled, _ = get_test_files_with_labels(dataset_name)
        if not test_files_labeled:
            print("No labeled test files found (need imagesTs/ + labelsTs/)!")
            sys.exit(1)

        print("\n=== Inference mode ===")
        print(f"Labeled test files: {len(test_files_labeled)}")

        labeled_test_ds = TrainDataset(
            test_files_labeled, transform=test_transforms, label_transform=test_transforms
        )
        labeled_test_loader = DataLoader(
            labeled_test_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]
        )
        labeled_test_loader = fabric.setup_dataloaders(labeled_test_loader)

        save_dir = (
            str(Path(args.output_dir) / dataset_name / model_name)
            if args.save_predictions
            else None
        )
        results = infer_with_metrics(
            fabric,
            model,
            labeled_test_loader,
            config["metrics"],
            out_channels,
            save_dir=save_dir,
            spatial_dims=spatial_dims,
        )

        # Log to W&B chart + summary with infer/ prefix
        wandb_metrics = {f"infer/{k}": v for k, v in results.items()}
        wandb.log(wandb_metrics)  # type: ignore[unresolved-attribute]
        for k, v in wandb_metrics.items():
            wandb.summary[k] = v  # type: ignore[unresolved-attribute]

        summary_parts = [f"{k}: {v:.4f}" for k, v in results.items()]
        print(f"\nInference results: {' - '.join(summary_parts)}")
        if save_dir:
            print(f"Predictions saved to: {save_dir}")
        wandb.finish()  # type: ignore[unresolved-attribute]
        sys.exit(0)

    # Setup learning rate scheduler
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif config["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, config["epochs"] // 3), gamma=0.1
        )
    elif config["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=max(1, config["patience"] // 2)
        )
    else:
        scheduler = None

    no_improve_count = 0

    total_epochs = config["epochs"]
    remaining_epochs = max(0, total_epochs - start_epoch)
    if resume_from:
        print(
            f"\nResuming training from epoch {start_epoch}/{total_epochs} ({remaining_epochs} epochs remaining)..."
        )
    else:
        print(f"\nTraining for {total_epochs} epoch(s)...")

    best_loss = float("inf")
    for epoch in range(remaining_epochs):
        current_epoch = start_epoch + epoch + 1
        result = train_one_epoch(fabric, model, train_loader, loss_fn, optimizer, config["metrics"])
        loss = result["loss"]

        epoch_metrics = {"loss": loss}
        if "dice" in result:
            epoch_metrics["dice"] = result["dice"]
        if "iou" in result:
            epoch_metrics["iou"] = result["iou"]

        wandb.log({"epoch": current_epoch, **epoch_metrics})  # type: ignore[unresolved-attribute]

        is_best = loss < best_loss
        if is_best:
            best_loss = loss

        logger.save_checkpoint(model, current_epoch - 1, optimizer, is_best=is_best)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        if config["early_stopping_enabled"]:
            if not is_best:
                no_improve_count += 1
                if no_improve_count >= config["patience"]:
                    print(
                        f"Early stopping triggered after {current_epoch} epochs (no improvement for {config['patience']} epochs)."
                    )
                    break
            else:
                no_improve_count = 0

        log_msg = f"Epoch {current_epoch}/{total_epochs} - Loss: {loss:.4f}"
        if "dice" in result:
            log_msg += f" - Dice: {result['dice']:.4f}"
        if "iou" in result:
            log_msg += f" - IoU: {result['iou']:.4f}"
        print(log_msg)

    wandb.finish()  # type: ignore[unresolved-attribute]

    if args.save_predictions:
        save_dir = Path(args.output_dir) / dataset_name / model_name
        print(f"\nRunning inference on {len(test_files)} test samples...")
        print(f"Saving predictions to: {save_dir}")
        infer(fabric, model, test_loader, str(save_dir), spatial_dims)

    print(f"\nCheckpoints saved to: {logger.run_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
