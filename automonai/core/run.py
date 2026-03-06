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
    split_train_val,
    TrainDataset,
    TestDataset,
    DictTransform,
)
from .models import get_model  # noqa: E402
from .inference import infer, infer_with_metrics, ensemble_infer_with_metrics  # noqa: E402
from .results import RunLogger  # noqa: E402
from .train import train_one_epoch, validate, get_loss  # noqa: E402
from .transforms import get_transforms  # noqa: E402
from .inferers import get_inferer  # noqa: E402


def _create_optimizer(name, parameters, lr):
    """Create optimizer by name."""
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    elif name == "novograd":
        from monai.optimizers import Novograd

        return Novograd(parameters, lr=lr)
    elif name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr)
    else:  # adam
        return torch.optim.Adam(parameters, lr=lr)


def _create_scheduler(name, optimizer, epochs, patience):
    """Create LR scheduler by name."""
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, patience // 2))
    elif name == "warmup_cosine":
        from monai.optimizers.lr_scheduler import WarmupCosineSchedule

        return WarmupCosineSchedule(optimizer, warmup_steps=max(1, epochs // 10), t_total=epochs)
    elif name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, epochs // 4), T_mult=2
        )
    elif name == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=1.0)
    else:
        return None


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Convert string booleans to actual bools
    args.augment = args.augment == "true"
    args.early_stopping = args.early_stopping == "true"

    # Filter out "none" from list args
    args.norm = [v for v in args.norm if v != "none"]
    args.crop = [v for v in args.crop if v != "none"]
    args.extra_transforms = [v for v in args.extra_transforms if v != "none"]

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
        inference_cache_rate = loaded_config.get("inference_cache_rate", cache_rate)
        inference_cache_dir = loaded_config.get("inference_cache_dir", cache_dir)
        img_size = loaded_config.get("image_size", args.img_size)
    else:
        model_name = args.model
        train_dataset_class = args.train_dataset_class
        inference_dataset_class = args.inference_dataset_class
        cache_rate = args.cache_rate
        smart_replace_rate = args.smart_replace_rate
        cache_dir = args.cache_dir
        inference_cache_rate = (
            args.inference_cache_rate if args.inference_cache_rate is not None else cache_rate
        )
        inference_cache_dir = (
            args.inference_cache_dir if args.inference_cache_dir is not None else cache_dir
        )
        img_size = args.img_size

    print(f"\nSelected model: {model_name}")
    print(f"  In channels: {in_channels}, Out channels: {out_channels}")

    train_files, is_3d = get_train_files(dataset_name)
    test_files, _ = get_test_files(dataset_name)

    spatial_dims = 3 if is_3d else 2
    print(f"  Detected: {'3D' if is_3d else '2D'} dataset")

    # Auto-infer val_split=kfold when cross_val and cv_fold are both set
    val_split = args.val_split
    if val_split == "none" and args.cross_val is not None and args.cv_fold is not None:
        val_split = "kfold"

    # Split train files into train/val
    n_folds = args.cross_val or 5
    fold_idx = (args.cv_fold or 1) - 1  # convert 1-based to 0-based
    train_files, val_files = split_train_val(
        train_files,
        mode=val_split,
        val_ratio=args.val_ratio,
        n_folds=n_folds,
        fold=fold_idx,
        seed=args.split_seed,
        split_file=args.split_file,
    )

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    if val_split != "none":
        print(f"Val split: {val_split}" + (f" (fold {args.cv_fold}/{n_folds})" if val_split == "kfold" else ""))

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
        config["extra_transforms"] = loaded_config.get("extra_transforms", [])
        config["optimizer"] = args.optimizer
        config["scheduler"] = args.scheduler
        config["mixed_precision"] = args.mixed_precision
        config["early_stopping_enabled"] = args.early_stopping
        config["patience"] = args.patience
        config["resume_from"] = resume_from
        config["resume_checkpoint"] = resume_checkpoint
        config["start_epoch"] = start_epoch
        config["inferer"] = loaded_config.get("inferer", args.inferer)
        config["deep_supervision"] = loaded_config.get("deep_supervision", args.deep_supervision)
        config["val_split"] = loaded_config.get("val_split", val_split)
        config["val_ratio"] = loaded_config.get("val_ratio", args.val_ratio)
        config["split_seed"] = loaded_config.get("split_seed", args.split_seed)
        config["best_metric"] = loaded_config.get("best_metric", args.best_metric)
        config["val_interval"] = loaded_config.get("val_interval", args.val_interval)
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
            "extra_transforms": args.extra_transforms,
            "optimizer": args.optimizer,
            "mixed_precision": args.mixed_precision,
            "scheduler": args.scheduler,
            "early_stopping_enabled": args.early_stopping,
            "patience": args.patience,
            "inferer": args.inferer,
            "deep_supervision": args.deep_supervision,
            "val_split": val_split,
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
            "best_metric": args.best_metric,
            "val_interval": args.val_interval,
        }
    if resume_from:
        config["resume_from"] = resume_from
        config["resume_checkpoint"] = resume_checkpoint
        config["start_epoch"] = start_epoch

    # Save config to run directory for future resume (skip for inference to preserve training config)
    if args.mode != "infer" or not resume_from:
        logger.save_config(config)
        # Save split info for reproducibility
        if val_split != "none":
            import json as _json
            split_info = {
                "mode": val_split,
                "train": [f["image"] for f in train_files],
                "val": [f["image"] for f in val_files],
            }
            with open(logger.run_dir / "split.json", "w") as f:
                _json.dump(split_info, f, indent=2)
    wandb_kwargs = {"project": "AutoMONAI", "config": config, "dir": str(Path.home() / ".wandb")}
    wandb_kwargs["name"] = args.run_id or f"{dataset_name}_{model_name}"
    if args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)  # type: ignore[unresolved-attribute]
    print(f"W&B run ID: {wandb.run.id}")  # type: ignore[unresolved-attribute]

    # Initialize Fabric for device and distributed training management
    _precision_map = {"fp16": "16-mixed", "bf16": "bf16-mixed"}
    precision = _precision_map.get(args.mixed_precision) if args.mixed_precision != "no" else None
    fabric = Fabric(accelerator="auto" if not args.device else args.device, precision=precision)  # type: ignore[invalid-argument-type]
    fabric.launch()

    print(f"Using device: {fabric.device}")

    extra_t = config.get("extra_transforms", [])
    train_transforms = get_transforms(
        img_size,
        spatial_dims,
        norm=args.norm,
        crop=args.crop,
        is_train=True,
        extra_transforms=extra_t if extra_t else None,
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

    # Create val_loader if we have validation files
    val_loader = None
    if val_files:
        val_ds = TrainDataset(val_files, transform=test_transforms, label_transform=test_transforms)
        val_loader = DataLoader(
            val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]
        )

    if inference_dataset_class == "Dataset":
        test_ds = TestDataset(test_files, transform=test_transforms)
    else:
        test_ds = create_inference_dataset(
            inference_dataset_class,
            test_files,
            test_transforms,
            cache_rate=inference_cache_rate,
            cache_dir=inference_cache_dir,
        )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]
    )

    deep_sup = config.get("deep_supervision") == "true"
    model = get_model(
        model_name,
        in_channels,
        out_channels,
        spatial_dims,
        config["image_size"],
        deep_supervision=deep_sup,
    )
    loss_fn = get_loss(config["loss"], deep_supervision=deep_sup)

    # Create optimizer
    optimizer = _create_optimizer(config["optimizer"], model.parameters(), config["learning_rate"])

    # Setup model, optimizer, and dataloaders with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    if val_loader is not None:
        train_loader, val_loader, test_loader = fabric.setup_dataloaders(
            train_loader, val_loader, test_loader
        )
    else:
        train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Load checkpoint if resuming
    if resume_from:
        checkpoint_path = RunLogger.get_checkpoint_path(resume_from, resume_checkpoint)
        checkpoint_data = RunLogger.load_checkpoint(str(checkpoint_path))
        model.load_state_dict(checkpoint_data["model_state"])
        if checkpoint_data["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")

    # Setup inferer
    roi_size = (config["image_size"],) * spatial_dims
    inferer = get_inferer(
        config.get("inferer", "simple"), model=model, roi_size=roi_size, spatial_dims=spatial_dims
    )
    if inferer:
        print(f"Using inferer: {config.get('inferer', 'simple')}")

    # Ensemble inference mode
    if args.ensemble_folds and args.fold_dirs:
        test_files_labeled, _ = get_test_files_with_labels(dataset_name)
        if not test_files_labeled:
            print("No labeled test files found (need imagesTs/ + labelsTs/)!")
            sys.exit(1)

        print("\n=== Ensemble inference mode ===")
        print(f"Method: {args.ensemble_method}")
        print(f"Fold dirs: {len(args.fold_dirs)}")
        print(f"Labeled test files: {len(test_files_labeled)}")

        labeled_test_ds = TrainDataset(
            test_files_labeled, transform=test_transforms, label_transform=test_transforms
        )
        labeled_test_loader = DataLoader(
            labeled_test_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]
        )
        labeled_test_loader = fabric.setup_dataloaders(labeled_test_loader)

        # Load K models one at a time
        ensemble_models = []
        for fold_dir in args.fold_dirs:
            fold_ckpt = Path(fold_dir) / "checkpoints" / "best_model.pt"
            if not fold_ckpt.exists():
                print(f"Warning: no best_model.pt in {fold_dir}, skipping")
                continue
            fold_model = get_model(
                model_name, in_channels, out_channels, spatial_dims, config["image_size"],
                deep_supervision=deep_sup,
            )
            fold_model = fabric.setup(fold_model)
            ckpt_data = RunLogger.load_checkpoint(str(fold_ckpt))
            fold_model.load_state_dict(ckpt_data["model_state"])
            ensemble_models.append(fold_model)

        if not ensemble_models:
            print("No valid fold models found!")
            sys.exit(1)

        save_dir = str(Path(args.output_dir) / dataset_name / model_name / "ensemble")
        wandb.define_metric("ensemble/*", step_metric="ensemble_step")  # type: ignore[unresolved-attribute]
        _ens_step = [0]

        def _wandb_ens_log(metrics):
            metrics["ensemble_step"] = _ens_step[0]
            _ens_step[0] += 1
            wandb.log(metrics)  # type: ignore[unresolved-attribute]

        results = ensemble_infer_with_metrics(
            fabric,
            ensemble_models,
            labeled_test_loader,
            config["metrics"],
            out_channels,
            save_dir=save_dir,
            spatial_dims=spatial_dims,
            inferer=inferer,
            method=args.ensemble_method,
            wandb_log=_wandb_ens_log,
        )

        for k, v in results.items():
            wandb.summary[f"ensemble/{k}"] = v  # type: ignore[unresolved-attribute]

        summary_parts = [f"{k}: {v:.4f}" for k, v in results.items()]
        print(f"\nEnsemble results: {' - '.join(summary_parts)}")
        print(f"Predictions saved to: {save_dir}")
        wandb.finish()  # type: ignore[unresolved-attribute]
        sys.exit(0)

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
        wandb.define_metric("infer/*", step_metric="infer_step")  # type: ignore[unresolved-attribute]
        _infer_step = [0]

        def _wandb_infer_log(metrics):
            metrics["infer_step"] = _infer_step[0]
            _infer_step[0] += 1
            wandb.log(metrics)  # type: ignore[unresolved-attribute]

        results = infer_with_metrics(
            fabric,
            model,
            labeled_test_loader,
            config["metrics"],
            out_channels,
            save_dir=save_dir,
            spatial_dims=spatial_dims,
            inferer=inferer,
            wandb_log=_wandb_infer_log,
        )

        # Log aggregate inference results to W&B summary
        for k, v in results.items():
            wandb.summary[f"infer/{k}"] = v  # type: ignore[unresolved-attribute]

        summary_parts = [f"{k}: {v:.4f}" for k, v in results.items()]
        print(f"\nInference results: {' - '.join(summary_parts)}")
        if save_dir:
            print(f"Predictions saved to: {save_dir}")
        wandb.finish()  # type: ignore[unresolved-attribute]
        sys.exit(0)

    # Setup learning rate scheduler
    scheduler = _create_scheduler(
        config["scheduler"], optimizer, config["epochs"], config["patience"]
    )

    no_improve_count = 0

    total_epochs = config["epochs"]
    remaining_epochs = max(0, total_epochs - start_epoch)
    if resume_from:
        print(
            f"\nResuming training from epoch {start_epoch}/{total_epochs} ({remaining_epochs} epochs remaining)..."
        )
    else:
        print(f"\nTraining for {total_epochs} epoch(s)...")

    # Best metric tracking
    best_metric_name = config.get("best_metric", "val_loss")
    # Fall back to train_loss when no val data
    if val_loader is None and best_metric_name.startswith("val_"):
        best_metric_name = "train_loss"
    # For loss: lower is better; for dice/iou: higher is better
    higher_is_better = best_metric_name in ("val_dice", "val_iou")
    best_metric_value = float("-inf") if higher_is_better else float("inf")
    val_interval = config.get("val_interval", 1)

    for epoch in range(remaining_epochs):
        current_epoch = start_epoch + epoch + 1
        result = train_one_epoch(
            fabric,
            model,
            train_loader,
            loss_fn,
            optimizer,
            config["metrics"],
            num_classes=out_channels,
        )
        loss = result["loss"]

        epoch_metrics = {"loss": loss}
        for metric_name in config["metrics"]:
            if metric_name in result:
                epoch_metrics[metric_name] = result[metric_name]

        # Run validation if val_loader exists and interval matches
        val_result = {}
        if val_loader is not None and current_epoch % val_interval == 0:
            val_result = validate(
                fabric, model, val_loader, loss_fn,
                metrics=config["metrics"], num_classes=out_channels,
            )
            epoch_metrics.update(val_result)

        wandb.log({"epoch": current_epoch, **epoch_metrics})  # type: ignore[unresolved-attribute]

        # Best model selection based on configured metric
        if best_metric_name == "train_loss":
            tracked_value = loss
        elif best_metric_name in val_result:
            tracked_value = val_result[best_metric_name]
        elif best_metric_name == "val_loss" and "val_loss" in val_result:
            tracked_value = val_result["val_loss"]
        else:
            tracked_value = loss  # fallback

        if higher_is_better:
            is_best = tracked_value > best_metric_value
        else:
            is_best = tracked_value < best_metric_value
        if is_best:
            best_metric_value = tracked_value

        logger.save_checkpoint(model, current_epoch - 1, optimizer, is_best=is_best)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(tracked_value)
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
        for metric_name in config["metrics"]:
            if metric_name in result:
                log_msg += f" - {metric_name}: {result[metric_name]:.4f}"
        if val_result:
            for k, v in val_result.items():
                log_msg += f" - {k}: {v:.4f}"
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
