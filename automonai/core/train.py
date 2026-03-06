import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceLoss, FocalLoss
import torch.nn as nn
from lightning.fabric import Fabric


def get_loss(loss_name, deep_supervision=False):
    if loss_name == "dice":
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_name == "focal":
        loss_fn = FocalLoss(to_onehot_y=True, use_softmax=True)
    elif loss_name == "dice_ce":
        from monai.losses import DiceCELoss

        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    elif loss_name == "dice_focal":
        from monai.losses import DiceFocalLoss

        loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "generalized_dice":
        from monai.losses import GeneralizedDiceLoss

        loss_fn = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "generalized_wasserstein_dice":
        from monai.losses import GeneralizedWassersteinDiceLoss

        # Default 2x2 distance matrix; user should customise for their label count
        dist_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss_fn = GeneralizedWassersteinDiceLoss(dist_matrix=dist_matrix)
    elif loss_name == "generalized_dice_focal":
        from monai.losses import GeneralizedDiceFocalLoss

        loss_fn = GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "tversky":
        from monai.losses import TverskyLoss

        loss_fn = TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.3, beta=0.7)
    elif loss_name == "hausdorff_dt":
        from monai.losses import HausdorffDTLoss

        loss_fn = HausdorffDTLoss(softmax=True)
    elif loss_name == "log_hausdorff_dt":
        from monai.losses import LogHausdorffDTLoss

        loss_fn = LogHausdorffDTLoss(softmax=True)
    elif loss_name == "soft_cl_dice":
        from monai.losses import SoftclDiceLoss

        loss_fn = SoftclDiceLoss(softmax=True)
    elif loss_name == "soft_dice_cl_dice":
        from monai.losses import SoftDiceclDiceLoss

        loss_fn = SoftDiceclDiceLoss(softmax=True)
    elif loss_name == "masked_dice":
        from monai.losses import MaskedDiceLoss

        loss_fn = MaskedDiceLoss(softmax=True)
    elif loss_name == "nacl":
        from monai.losses import NACLLoss

        loss_fn = NACLLoss(classes=2, kernel_size=3)
    elif loss_name == "asymmetric_unified_focal":
        from monai.losses import AsymmetricUnifiedFocalLoss

        loss_fn = AsymmetricUnifiedFocalLoss()
    elif loss_name == "ssim":
        from monai.losses import SSIMLoss

        loss_fn = SSIMLoss(spatial_dims=2)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    if deep_supervision:
        from monai.losses import DeepSupervisionLoss

        return DeepSupervisionLoss(loss_fn)

    return loss_fn


def get_metrics(metric_names, num_classes=2):
    metrics = {}
    if "dice" in metric_names:
        metrics["dice"] = DiceMetric(include_background=False, reduction="mean")
    if "iou" in metric_names:
        metrics["iou"] = MeanIoU(include_background=False, reduction="mean")
    if "hausdorff_distance" in metric_names or "hausdorff" in metric_names:
        from monai.metrics import HausdorffDistanceMetric

        key = "hausdorff_distance" if "hausdorff_distance" in metric_names else "hausdorff"
        metrics[key] = HausdorffDistanceMetric(include_background=False, reduction="mean")
    if "surface_distance" in metric_names:
        from monai.metrics import SurfaceDistanceMetric

        metrics["surface_distance"] = SurfaceDistanceMetric(
            include_background=False, reduction="mean"
        )
    if "surface_dice" in metric_names:
        from monai.metrics import SurfaceDiceMetric

        # One threshold per foreground class
        n_foreground = max(num_classes - 1, 1)
        metrics["surface_dice"] = SurfaceDiceMetric(
            class_thresholds=[2.0] * n_foreground, include_background=False, reduction="mean"
        )
    if "generalized_dice" in metric_names:
        from monai.metrics import GeneralizedDiceScore

        metrics["generalized_dice"] = GeneralizedDiceScore(
            include_background=False, reduction="mean"
        )
    if "confusion_matrix" in metric_names:
        from monai.metrics import ConfusionMatrixMetric

        metrics["confusion_matrix"] = ConfusionMatrixMetric(
            include_background=False, metric_name="f1 score", reduction="mean"
        )
    if "rocauc" in metric_names:
        # ROC AUC needs special handling; store placeholder
        metrics["rocauc"] = None
    if "fbeta" in metric_names:
        from monai.metrics import FBetaScore

        metrics["fbeta"] = FBetaScore(include_background=False, reduction="mean")
    if "panoptic_quality" in metric_names:
        from monai.metrics import PanopticQualityMetric

        if num_classes == 2:
            metrics["panoptic_quality"] = PanopticQualityMetric(num_classes=num_classes)
        else:
            # PanopticQualityMetric only supports binary (2-channel) input
            pass
    if "calibration" in metric_names:
        # Calibration metric needs special handling
        metrics["calibration"] = None
    return metrics


def compute_metrics(metrics, y_pred, y):
    results = {}
    for name, metric in metrics.items():
        if metric is not None:
            metric(y_pred, y)
    return results


def get_metric_values(metrics):
    results = {}
    for name, metric in metrics.items():
        if metric is None:
            continue
        val = metric.aggregate()
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) == 1 else val
        if isinstance(val, torch.Tensor):
            val = val.item() if val.numel() == 1 else val.mean().item()
        if isinstance(val, (list, tuple)):
            val = sum(
                v
                if isinstance(v, (int, float))
                else v.mean().item()
                if isinstance(v, torch.Tensor)
                else 0
                for v in val
            ) / max(len(val), 1)
        results[name] = val
        metric.reset()
    return results


def train_one_epoch(
    fabric: Fabric, model, train_loader, loss_fn, optimizer, metrics=None, num_classes=2
):
    model.train()
    total_loss = 0

    metric_aggregators = {}
    if metrics:
        metric_aggregators = get_metrics(metrics, num_classes=num_classes)

    for batch in train_loader:
        inputs = batch["image"]
        labels = batch["label"]

        optimizer.zero_grad()
        outputs = model(inputs)

        # Deep supervision: pass all outputs to DeepSupervisionLoss
        from monai.losses import DeepSupervisionLoss

        if isinstance(loss_fn, DeepSupervisionLoss) and isinstance(outputs, (tuple, list)):
            loss = loss_fn(outputs, labels.float())
            main_output = outputs[0]
        else:
            # Handle tuple outputs (e.g. from VAE models)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            main_output = outputs

            # CrossEntropyLoss expects targets without channel dimension
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(main_output, labels.squeeze(1).long())
            else:
                loss = loss_fn(main_output, labels.long())

        fabric.backward(loss)
        optimizer.step()

        total_loss += loss.item()

        if metric_aggregators:
            with torch.no_grad():
                num_classes = main_output.shape[1]
                preds = torch.softmax(main_output, dim=1)
                preds_onehot = (preds == preds.max(dim=1, keepdim=True).values).float()
                labels_onehot = F.one_hot(labels.squeeze(1).long(), num_classes)
                # Permute to (B, C, ...) format
                dims = list(range(labels_onehot.ndim))
                dims = [dims[0], dims[-1]] + dims[1:-1]
                labels_onehot = labels_onehot.permute(*dims).float()
            compute_metrics(metric_aggregators, preds_onehot, labels_onehot)

    avg_loss = total_loss / len(train_loader)
    result = {"loss": avg_loss}

    if metric_aggregators:
        metric_values = get_metric_values(metric_aggregators)
        result.update(metric_values)

    return result


def validate(fabric: Fabric, model, val_loader, loss_fn, metrics=None, num_classes=2):
    """No-grad forward pass on val set. Returns {val_loss, val_<metric>: value}."""
    model.eval()
    total_loss = 0

    metric_aggregators = {}
    if metrics:
        metric_aggregators = get_metrics(metrics, num_classes=num_classes)

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"]
            labels = batch["label"]

            outputs = model(inputs)

            from monai.losses import DeepSupervisionLoss

            if isinstance(loss_fn, DeepSupervisionLoss) and isinstance(outputs, (tuple, list)):
                loss = loss_fn(outputs, labels.float())
                main_output = outputs[0]
            else:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                main_output = outputs

                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss = loss_fn(main_output, labels.squeeze(1).long())
                else:
                    loss = loss_fn(main_output, labels.long())

            total_loss += loss.item()

            if metric_aggregators:
                num_classes = main_output.shape[1]
                preds = torch.softmax(main_output, dim=1)
                preds_onehot = (preds == preds.max(dim=1, keepdim=True).values).float()
                labels_onehot = F.one_hot(labels.squeeze(1).long(), num_classes)
                dims = list(range(labels_onehot.ndim))
                dims = [dims[0], dims[-1]] + dims[1:-1]
                labels_onehot = labels_onehot.permute(*dims).float()
                compute_metrics(metric_aggregators, preds_onehot, labels_onehot)

    model.train()

    avg_loss = total_loss / max(len(val_loader), 1)
    result = {"val_loss": avg_loss}

    if metric_aggregators:
        metric_values = get_metric_values(metric_aggregators)
        for k, v in metric_values.items():
            result[f"val_{k}"] = v

    return result
