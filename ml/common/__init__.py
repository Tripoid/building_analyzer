"""Common abstractions shared across all ML modules."""

from ml.common.base_dataset import BaseDataset
from ml.common.base_model import (
    BaseClassificationModel,
    BaseDetectionModel,
    BaseSegmentationModel,
)
from ml.common.metrics import (
    compute_iou,
    compute_map,
    compute_pixel_accuracy,
    compute_top_k_accuracy,
)
from ml.common.registry import ModelRegistry
from ml.common.transforms import get_classification_transforms, get_segmentation_transforms

__all__ = [
    "BaseDataset",
    "BaseSegmentationModel",
    "BaseDetectionModel",
    "BaseClassificationModel",
    "ModelRegistry",
    "get_segmentation_transforms",
    "get_classification_transforms",
    "compute_iou",
    "compute_pixel_accuracy",
    "compute_map",
    "compute_top_k_accuracy",
]
