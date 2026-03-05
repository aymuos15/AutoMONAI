import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceLoss, FocalLoss
import torch.nn as nn
from lightning.fabric import Fabric


def get_loss(loss_name):
    if loss_name == "dice":
        return DiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "focal":
        return FocalLoss(to_onehot_y=True, use_softmax=True)
    elif loss_name == "dice_ce":
        from monai.losses import DiceCELoss

        return DiceCELoss(to_onehot_y=True, softmax=True)
    elif loss_name == "dice_focal":
        from monai.losses import DiceFocalLoss

        return DiceFocalLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "generalized_dice":
        from monai.losses import GeneralizedDiceLoss

        return GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "generalized_wasserstein_dice":
        from monai.losses import GeneralizedWassersteinDiceLoss

        # Default 2x2 distance matrix; user should customise for their label count
        dist_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        return GeneralizedWassersteinDiceLoss(dist_matrix=dist_matrix)
    elif loss_name == "generalized_dice_focal":
        from monai.losses import GeneralizedDiceFocalLoss

        return GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "tversky":
        from monai.losses import TverskyLoss

        return TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.3, beta=0.7)
    elif loss_name == "hausdorff_dt":
        from monai.losses import HausdorffDTLoss

        return HausdorffDTLoss(softmax=True)
    elif loss_name == "log_hausdorff_dt":
        from monai.losses import LogHausdorffDTLoss

        return LogHausdorffDTLoss(softmax=True)
    elif loss_name == "soft_cl_dice":
        from monai.losses import SoftclDiceLoss

        return SoftclDiceLoss(softmax=True)
    elif loss_name == "soft_dice_cl_dice":
        from monai.losses import SoftDiceclDiceLoss

        return SoftDiceclDiceLoss(softmax=True)
    elif loss_name == "masked_dice":
        from monai.losses import MaskedDiceLoss

        return MaskedDiceLoss(softmax=True)
    elif loss_name == "nacl":
        from monai.losses import NACLLoss

        return NACLLoss(classes=2, kernel_size=3)
    elif loss_name == "asymmetric_unified_focal":
        from monai.losses import AsymmetricUnifiedFocalLoss

        return AsymmetricUnifiedFocalLoss()
    elif loss_name == "ssim":
        from monai.losses import SSIMLoss

        return SSIMLoss(spatial_dims=2)
    elif loss_name == "deep_supervision":
        from monai.losses import DeepSupervisionLoss

        return DeepSupervisionLoss(DiceLoss(to_onehot_y=True, softmax=True))
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def get_metrics(metric_names):
    metrics = {}
    if "dice" in metric_names:
        metrics["dice"] = DiceMetric(include_background=False, reduction="mean")
    if "iou" in metric_names:
        metrics["iou"] = MeanIoU(include_background=False, reduction="mean")
    if "hausdorff" in metric_names:
        from monai.metrics import HausdorffDistanceMetric

        metrics["hausdorff"] = HausdorffDistanceMetric(include_background=False, reduction="mean")
    if "surface_distance" in metric_names:
        from monai.metrics import SurfaceDistanceMetric

        metrics["surface_distance"] = SurfaceDistanceMetric(
            include_background=False, reduction="mean"
        )
    if "surface_dice" in metric_names:
        from monai.metrics import SurfaceDiceMetric

        metrics["surface_dice"] = SurfaceDiceMetric(
            class_thresholds=[2.0], include_background=False, reduction="mean"
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

        metrics["panoptic_quality"] = PanopticQualityMetric()
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
        if isinstance(val, torch.Tensor):
            val = val.item() if val.numel() == 1 else val.mean().item()
        results[name] = val
        metric.reset()
    return results


def train_one_epoch(fabric: Fabric, model, train_loader, loss_fn, optimizer, metrics=None):
    model.train()
    total_loss = 0

    metric_aggregators = {}
    if metrics:
        metric_aggregators = get_metrics(metrics)

    for batch in train_loader:
        inputs = batch["image"]
        labels = batch["label"]

        optimizer.zero_grad()
        outputs = model(inputs)

        # Handle tuple outputs (e.g. from VAE models or deep supervision)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        # CrossEntropyLoss expects targets without channel dimension
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(outputs, labels.squeeze(1).long())
        else:
            loss = loss_fn(outputs, labels.long())

        fabric.backward(loss)
        optimizer.step()

        total_loss += loss.item()

        if metric_aggregators:
            with torch.no_grad():
                num_classes = outputs.shape[1]
                preds = torch.softmax(outputs, dim=1)
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
