"""Configuration API routes for models and datasets."""

from fastapi import APIRouter
from src.config import DATASETS, MODELS

router = APIRouter()


@router.get("/api/models")
async def get_models():
    """Get available models."""
    return MODELS


@router.get("/api/datasets")
async def get_datasets():
    """Get available datasets."""
    return DATASETS
