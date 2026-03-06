"""Facade segmentation module.

Detects structural facade elements (wall, window, door, balcony, cornice) and
separates intact from damaged segments.
"""

from ml.facade_segmentation.dataset import FacadeSegmentationDataset
from ml.facade_segmentation.inference import SegmentationInferencer
from ml.facade_segmentation.model import DeepLabV3PlusSegmentation, SegmentationModel
from ml.facade_segmentation.train import SegmentationTrainer

__all__ = [
    "FacadeSegmentationDataset",
    "SegmentationModel",
    "DeepLabV3PlusSegmentation",
    "SegmentationTrainer",
    "SegmentationInferencer",
]
