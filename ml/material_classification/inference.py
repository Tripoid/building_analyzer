"""Material classification inferencer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ml.common.base_model import BaseClassificationModel, ClassificationResult
from ml.common.registry import ModelRegistry
from ml.material_classification.utils import extract_region_crops, aggregate_region_materials

logger = logging.getLogger(__name__)


@dataclass
class MaterialInferencerConfig:
    """Configuration for the material classification inferencer."""

    model_name: str = "cnn_classifier"
    model_weights: str | None = None
    device: str = "auto"
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 16
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class MaterialInferencer:
    """Runs material classification inference on image crops.

    Supports:
    - Single-crop inference via :meth:`predict_from_array`.
    - Batch crop inference via :meth:`predict_batch`.
    - Region-based inference via :meth:`predict_regions`, which accepts an
      image and a list of bounding boxes and returns per-region materials.

    Args:
        model: A loaded :class:`~ml.common.base_model.BaseClassificationModel`.
        config: Inferencer configuration.
    """

    def __init__(
        self,
        model: BaseClassificationModel,
        config: MaterialInferencerConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = config or MaterialInferencerConfig()
        self.device = self.cfg.resolve_device()
        self.model.to(self.device)
        self.model.eval()

        h, w = self.cfg.image_size
        self._transform: A.Compose = A.Compose(
            [
                A.Resize(h, w),
                A.Normalize(mean=self.cfg.mean, std=self.cfg.std),
                ToTensorV2(),
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_path(self, path: str | Path) -> ClassificationResult:
        """Load an image from *path* and classify it as a whole.

        Args:
            path: Path to an image file.

        Returns:
            A :class:`~ml.common.base_model.ClassificationResult`.
        """
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return self.predict_from_array(image)

    def predict_from_array(self, image: np.ndarray) -> ClassificationResult:
        """Classify a single HWC uint8 NumPy image.

        Args:
            image: RGB array of shape (H, W, 3).

        Returns:
            A :class:`~ml.common.base_model.ClassificationResult`.
        """
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        results = self.model.predict(tensor)
        return results[0]

    def predict_batch(self, images: list[np.ndarray]) -> list[ClassificationResult]:
        """Classify a batch of HWC uint8 NumPy images.

        Automatically chunks into batches of size ``config.batch_size``.

        Args:
            images: List of RGB arrays.

        Returns:
            List of :class:`~ml.common.base_model.ClassificationResult`, one per image.
        """
        all_results: list[ClassificationResult] = []
        bs = self.cfg.batch_size
        for start in range(0, len(images), bs):
            chunk = images[start : start + bs]
            tensors = torch.stack([self._preprocess(img) for img in chunk]).to(self.device)
            all_results.extend(self.model.predict(tensors))
        return all_results

    def predict_regions(
        self,
        image: np.ndarray,
        boxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Classify materials in each bounding-box region of *image*.

        Args:
            image: Full RGB image array of shape (H, W, 3).
            boxes: List of ``[x1, y1, x2, y2]`` bounding boxes in pixel coords.

        Returns:
            List of dicts, one per box, each containing:
            ``"box"``, ``"label"``, ``"label_name"``, ``"score"``, ``"scores"``.
        """
        if not boxes:
            return []

        crops = extract_region_crops(image, boxes, target_size=self.cfg.image_size)
        classification_results = self.predict_batch(crops)

        output = []
        for box, result in zip(boxes, classification_results):
            output.append(
                {
                    "box": box,
                    "label": result.label,
                    "label_name": result.label_name,
                    "score": result.score,
                    "scores": result.scores,
                }
            )
        return output

    def predict_mask_regions(
        self,
        image: np.ndarray,
        segmentation_mask: np.ndarray,
        class_ids: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Classify materials for each segment in a segmentation mask.

        Args:
            image: Full RGB image array of shape (H, W, 3).
            segmentation_mask: Integer mask of shape (H, W) with class indices.
            class_ids: Subset of class indices to classify.  If ``None``,
                all unique values in the mask are used.

        Returns:
            Dict mapping segment class id → classification result dict.
        """
        ids = class_ids or list(np.unique(segmentation_mask).tolist())
        output: dict[int, dict[str, Any]] = {}
        h, w = image.shape[:2]

        for cls_id in ids:
            region_mask = (segmentation_mask == cls_id)
            if not region_mask.any():
                continue

            # Tight crop around the region
            rows = np.where(region_mask.any(axis=1))[0]
            cols = np.where(region_mask.any(axis=0))[0]
            y1, y2 = int(rows.min()), int(rows.max()) + 1
            x1, x2 = int(cols.min()), int(cols.max()) + 1

            crop = image[y1:y2, x1:x2]
            result = self.predict_from_array(crop)
            output[cls_id] = {
                "box": [x1, y1, x2, y2],
                "label": result.label,
                "label_name": result.label_name,
                "score": result.score,
                "scores": result.scores,
            }
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        return self._transform(image=image)["image"]  # type: ignore[return-value]

    @classmethod
    def from_config(
        cls,
        config: MaterialInferencerConfig,
        **model_kwargs: Any,
    ) -> "MaterialInferencer":
        """Instantiate from *config*, building the model from the registry."""
        model: BaseClassificationModel = ModelRegistry.build(
            config.model_name, namespace="classification", **model_kwargs
        )
        if config.model_weights:
            model.load(config.model_weights, map_location=config.resolve_device())
        return cls(model=model, config=config)
