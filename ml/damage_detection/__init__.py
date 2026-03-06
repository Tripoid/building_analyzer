"""Damage detection module.

Detects and highlights damaged areas in building facade images, producing
bounding boxes, instance masks, and confidence scores per damage region.
"""

from ml.damage_detection.dataset import DamageDetectionDataset
from ml.damage_detection.inference import DamageDetectionInferencer
from ml.damage_detection.model import AnchorFreeDamageDetector, DamageDetectionModel
from ml.damage_detection.train import DamageDetectionTrainer

__all__ = [
    "DamageDetectionDataset",
    "DamageDetectionModel",
    "AnchorFreeDamageDetector",
    "DamageDetectionTrainer",
    "DamageDetectionInferencer",
]
