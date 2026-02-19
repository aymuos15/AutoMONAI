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
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def get_metrics(metric_names):
    metrics = {}
    if "dice" in metric_names:
        metrics["dice"] = DiceMetric(include_background=False, reduction="mean")
    if "iou" in metric_names:
        metrics["iou"] = MeanIoU(include_background=False, reduction="mean")
    return metrics


def compute_metrics(metrics, y_pred, y):
    results = {}
    for name, metric in metrics.items():
        metric(y_pred, y)
    return results


def get_metric_values(metrics):
    results = {}
    for name, metric in metrics.items():
        results[name] = metric.aggregate().item()
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

        # CrossEntropyLoss expects targets without channel dimension
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(outputs, labels.squeeze(1).long())
        else:
            loss = loss_fn(outputs, labels.long())

        fabric.backward(loss)
        optimizer.step()

        total_loss += loss.item()

        if metric_aggregators:
            compute_metrics(metric_aggregators, outputs, labels)

    avg_loss = total_loss / len(train_loader)
    result = {"loss": avg_loss}

    if metric_aggregators:
        metric_values = get_metric_values(metric_aggregators)
        result.update(metric_values)

    return result
