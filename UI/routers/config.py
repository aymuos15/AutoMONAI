"""Configuration API routes for models and datasets."""

from fastapi import APIRouter
from src.config import (
    DATASETS,
    MODELS,
    LOSSES_AVAILABLE,
    METRICS_AVAILABLE,
    OPTIMIZERS_AVAILABLE,
    SCHEDULERS_AVAILABLE,
    INFERERS_AVAILABLE,
    AUGMENTATION_TRANSFORMS,
    DATASET_CLASSES,
)

router = APIRouter()


@router.get("/api/models")
async def get_models():
    """Get available models."""
    return MODELS


@router.get("/api/datasets")
async def get_datasets():
    """Get available datasets."""
    return DATASETS


@router.get("/api/options")
async def get_options():
    """Get all available configuration options."""
    return {
        "losses": LOSSES_AVAILABLE,
        "metrics": METRICS_AVAILABLE,
        "optimizers": OPTIMIZERS_AVAILABLE,
        "schedulers": SCHEDULERS_AVAILABLE,
        "inferers": INFERERS_AVAILABLE,
        "augmentation_transforms": AUGMENTATION_TRANSFORMS,
        "dataset_classes": DATASET_CLASSES,
    }
