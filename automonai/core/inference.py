import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lightning.fabric import Fabric

from .train import get_metrics, compute_metrics, get_metric_values
from .inferers import run_inferer


def infer(fabric: Fabric, model, infer_loader, save_dir, spatial_dims=2, inferer=None):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(infer_loader):
            inputs = batch["image"]
            output = run_inferer(inferer, model, inputs)

            # Handle tuple outputs (e.g. from VAE models)
            if isinstance(output, (tuple, list)):
                output = output[0]

            result = torch.softmax(output, dim=1)
            result = torch.argmax(result, dim=1)

            result_np = result.cpu().numpy()[0]

            if spatial_dims == 3:
                import nibabel as nib

                out_path = os.path.join(save_dir, f"prediction_{idx:04d}.nii.gz")
                img = nib.Nifti1Image(result_np.astype(np.uint8), np.eye(4))
                nib.save(img, out_path)
            else:
                if result_np.ndim == 3:
                    result_np = result_np[result_np.shape[0] // 2]
                img = Image.fromarray((result_np * 255).astype("uint8"))
                img.save(os.path.join(save_dir, f"prediction_{idx:04d}.png"))

            print(
                f"Saved prediction_{idx:04d}.png"
                if spatial_dims == 2
                else f"Saved prediction_{idx:04d}.nii.gz"
            )


def infer_with_metrics(
    fabric: Fabric,
    model,
    test_loader,
    metric_names,
    num_classes,
    save_dir=None,
    spatial_dims=2,
    inferer=None,
    wandb_log=None,
):
    model.eval()
    metric_aggregators = get_metrics(metric_names, num_classes=num_classes)
    # Per-sample metrics for W&B charts
    per_sample_metrics = get_metrics(metric_names, num_classes=num_classes)
    total = len(test_loader)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs = batch["image"]
            labels = batch["label"]

            output = run_inferer(inferer, model, inputs)

            # Handle tuple outputs (e.g. from VAE models)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # One-hot encode predictions and labels for MONAI metrics
            preds = torch.softmax(output, dim=1)
            preds_onehot = (preds == preds.max(dim=1, keepdim=True).values).float()
            labels_onehot = F.one_hot(labels.squeeze(1).long(), num_classes)
            # Permute to (B, C, ...) format
            dims = list(range(labels_onehot.ndim))
            dims = [dims[0], dims[-1]] + dims[1:-1]
            labels_onehot = labels_onehot.permute(*dims).float()

            compute_metrics(metric_aggregators, preds_onehot, labels_onehot)

            # Log per-sample metrics to W&B for visible charts
            if wandb_log is not None:
                compute_metrics(per_sample_metrics, preds_onehot, labels_onehot)
                sample_results = get_metric_values(per_sample_metrics)
                wandb_log({f"infer/{k}": v for k, v in sample_results.items()})

            if save_dir:
                result = torch.argmax(preds, dim=1)
                result_np = result.cpu().numpy()[0]
                if spatial_dims == 3:
                    import nibabel as nib

                    out_path = os.path.join(save_dir, f"prediction_{idx:04d}.nii.gz")
                    img = nib.Nifti1Image(result_np.astype(np.uint8), np.eye(4))
                    nib.save(img, out_path)
                else:
                    if result_np.ndim == 3:
                        result_np = result_np[result_np.shape[0] // 2]
                    img = Image.fromarray((result_np * 255).astype("uint8"))
                    img.save(os.path.join(save_dir, f"prediction_{idx:04d}.png"))

            print(f"Inference {idx + 1}/{total}")

    results = get_metric_values(metric_aggregators)
    return results


def ensemble_infer_with_metrics(
    fabric,
    models,
    test_loader,
    metric_names,
    num_classes,
    save_dir=None,
    spatial_dims=2,
    inferer=None,
    method="mean",
    wandb_log=None,
):
    """Ensemble inference: average softmax or majority vote across K models."""
    for m in models:
        m.eval()

    metric_aggregators = get_metrics(metric_names, num_classes=num_classes)
    per_sample_metrics = get_metrics(metric_names, num_classes=num_classes)
    total = len(test_loader)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs = batch["image"]
            labels = batch["label"]

            if method == "mean":
                # Average softmax probabilities across models
                avg_probs = None
                for model in models:
                    output = run_inferer(inferer, model, inputs)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    probs = torch.softmax(output, dim=1)
                    if avg_probs is None:
                        avg_probs = probs
                    else:
                        avg_probs = avg_probs + probs
                avg_probs = avg_probs / len(models)
                preds_onehot = (avg_probs == avg_probs.max(dim=1, keepdim=True).values).float()
            else:  # vote
                # Majority vote: argmax per model, then mode
                votes = []
                for model in models:
                    output = run_inferer(inferer, model, inputs)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    probs = torch.softmax(output, dim=1)
                    votes.append(torch.argmax(probs, dim=1))
                stacked = torch.stack(votes, dim=0)  # (K, B, ...)
                ensemble_pred = torch.mode(stacked, dim=0).values  # (B, ...)
                preds_onehot = F.one_hot(ensemble_pred.long(), num_classes)
                dims = list(range(preds_onehot.ndim))
                dims = [dims[0], dims[-1]] + dims[1:-1]
                preds_onehot = preds_onehot.permute(*dims).float()
                avg_probs = preds_onehot  # for saving predictions

            labels_onehot = F.one_hot(labels.squeeze(1).long(), num_classes)
            dims = list(range(labels_onehot.ndim))
            dims = [dims[0], dims[-1]] + dims[1:-1]
            labels_onehot = labels_onehot.permute(*dims).float()

            compute_metrics(metric_aggregators, preds_onehot, labels_onehot)

            if wandb_log is not None:
                compute_metrics(per_sample_metrics, preds_onehot, labels_onehot)
                sample_results = get_metric_values(per_sample_metrics)
                wandb_log({f"ensemble/{k}": v for k, v in sample_results.items()})

            if save_dir:
                result = torch.argmax(avg_probs, dim=1)
                result_np = result.cpu().numpy()[0]
                if spatial_dims == 3:
                    import nibabel as nib
                    out_path = os.path.join(save_dir, f"prediction_{idx:04d}.nii.gz")
                    img = nib.Nifti1Image(result_np.astype(np.uint8), np.eye(4))
                    nib.save(img, out_path)
                else:
                    if result_np.ndim == 3:
                        result_np = result_np[result_np.shape[0] // 2]
                    img = Image.fromarray((result_np * 255).astype("uint8"))
                    img.save(os.path.join(save_dir, f"prediction_{idx:04d}.png"))

            print(f"Ensemble inference {idx + 1}/{total}")

    results = get_metric_values(metric_aggregators)
    return results
