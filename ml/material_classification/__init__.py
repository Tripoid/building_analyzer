"""Material classification module.

Identifies building materials (concrete, brick, glass, wood, metal, stone) in
arbitrary image crops, including damaged regions.
"""

from ml.material_classification.dataset import MaterialDataset
from ml.material_classification.inference import MaterialInferencer
from ml.material_classification.model import (
    CNNMaterialClassifier,
    MaterialClassifier,
    ViTMaterialClassifier,
)
from ml.material_classification.train import MaterialTrainer

__all__ = [
    "MaterialDataset",
    "MaterialClassifier",
    "CNNMaterialClassifier",
    "ViTMaterialClassifier",
    "MaterialTrainer",
    "MaterialInferencer",
]
