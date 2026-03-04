import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lightning.fabric import Fabric

from .train import get_metrics, compute_metrics, get_metric_values


def infer(fabric: Fabric, model, infer_loader, save_dir, spatial_dims=2):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(infer_loader):
            inputs = batch["image"]
            output = model(inputs)

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
    fabric: Fabric, model, test_loader, metric_names, num_classes, save_dir=None, spatial_dims=2
):
    model.eval()
    metric_aggregators = get_metrics(metric_names)
    total = len(test_loader)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs = batch["image"]
            labels = batch["label"]

            output = model(inputs)

            # One-hot encode predictions and labels for MONAI metrics
            preds = torch.softmax(output, dim=1)
            preds_onehot = (preds == preds.max(dim=1, keepdim=True).values).float()
            labels_onehot = F.one_hot(labels.squeeze(1).long(), num_classes)
            labels_onehot = labels_onehot.permute(0, 3, 1, 2).float()
            if spatial_dims == 3:
                labels_onehot = labels_onehot.permute(0, 1, 2, 3, 4)

            compute_metrics(metric_aggregators, preds_onehot, labels_onehot)

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
