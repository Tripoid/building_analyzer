"""Damage detection inferencer."""

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

from ml.common.base_model import BaseDetectionModel, DetectionResult
from ml.common.registry import ModelRegistry
from ml.damage_detection.utils import apply_nms, draw_detections

logger = logging.getLogger(__name__)


@dataclass
class DamageDetectionInferencerConfig:
    """Configuration for the damage detection inferencer."""

    model_name: str = "anchor_free"
    model_weights: str | None = None
    device: str = "auto"
    image_size: tuple[int, int] = (640, 640)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    score_threshold: float = 0.4
    nms_iou_threshold: float = 0.5
    max_detections: int = 100
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class DamageDetectionInferencer:
    """Runs damage detection inference on arbitrary images.

    Handles pre-processing, calls the model, applies NMS post-processing,
    and returns structured :class:`~ml.common.base_model.DetectionResult`
    objects.

    Args:
        model: A loaded :class:`~ml.common.base_model.BaseDetectionModel`.
        config: Inferencer configuration.
    """

    def __init__(
        self,
        model: BaseDetectionModel,
        config: DamageDetectionInferencerConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = config or DamageDetectionInferencerConfig()
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

    def predict_from_path(self, path: str | Path) -> "DamageDetectionPrediction":
        """Load an image from *path* and run inference.

        Args:
            path: Path to an image file.

        Returns:
            A :class:`DamageDetectionPrediction`.
        """
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return self.predict_from_array(image)

    def predict_from_array(self, image: np.ndarray) -> "DamageDetectionPrediction":
        """Run inference on a single HWC uint8 NumPy image.

        Args:
            image: RGB image array of shape (H, W, 3).

        Returns:
            A :class:`DamageDetectionPrediction`.
        """
        original_h, original_w = image.shape[:2]
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw = self.model(tensor)

        raw_pred = raw[0]
        boxes = raw_pred.get("boxes", torch.zeros(0, 4))
        labels = raw_pred.get("labels", torch.zeros(0, dtype=torch.long))
        scores = raw_pred.get("scores", torch.zeros(0))

        # NMS
        keep_boxes, keep_labels, keep_scores = apply_nms(
            boxes.cpu(),
            labels.cpu(),
            scores.cpu(),
            score_threshold=self.cfg.score_threshold,
            iou_threshold=self.cfg.nms_iou_threshold,
            max_detections=self.cfg.max_detections,
        )

        det_result = DetectionResult(instances=[])
        from ml.common.base_model import DetectionInstance
        for box, label, score in zip(keep_boxes, keep_labels, keep_scores):
            det_result.instances.append(
                DetectionInstance(
                    label=int(label.item()),
                    score=float(score.item()),
                    box=box.tolist(),
                )
            )

        return DamageDetectionPrediction(
            result=det_result,
            class_names=self.model.class_names,
            original_size=(original_h, original_w),
            model_size=self.cfg.image_size,
        )

    def predict_batch(self, images: list[np.ndarray]) -> list["DamageDetectionPrediction"]:
        """Run inference on a list of HWC uint8 NumPy images."""
        original_sizes = [img.shape[:2] for img in images]
        tensors = torch.stack(
            [self._preprocess(img) for img in images], dim=0
        ).to(self.device)

        with torch.no_grad():
            raw_list = self.model(tensors)

        predictions = []
        for raw_pred, (oh, ow) in zip(raw_list, original_sizes):
            boxes = raw_pred.get("boxes", torch.zeros(0, 4))
            labels = raw_pred.get("labels", torch.zeros(0, dtype=torch.long))
            scores = raw_pred.get("scores", torch.zeros(0))

            keep_boxes, keep_labels, keep_scores = apply_nms(
                boxes.cpu(),
                labels.cpu(),
                scores.cpu(),
                score_threshold=self.cfg.score_threshold,
                iou_threshold=self.cfg.nms_iou_threshold,
                max_detections=self.cfg.max_detections,
            )

            det_result = DetectionResult(instances=[])
            from ml.common.base_model import DetectionInstance
            for box, label, score in zip(keep_boxes, keep_labels, keep_scores):
                det_result.instances.append(
                    DetectionInstance(
                        label=int(label.item()),
                        score=float(score.item()),
                        box=box.tolist(),
                    )
                )
            predictions.append(
                DamageDetectionPrediction(
                    result=det_result,
                    class_names=self.model.class_names,
                    original_size=(oh, ow),
                    model_size=self.cfg.image_size,
                )
            )
        return predictions

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        return self._transform(image=image)["image"]  # type: ignore[return-value]

    @classmethod
    def from_config(
        cls,
        config: DamageDetectionInferencerConfig,
        **model_kwargs: Any,
    ) -> "DamageDetectionInferencer":
        """Instantiate from *config*, building the model from the registry."""
        model: BaseDetectionModel = ModelRegistry.build(
            config.model_name, namespace="detection", **model_kwargs
        )
        if config.model_weights:
            model.load(config.model_weights, map_location=config.resolve_device())
        return cls(model=model, config=config)


# ---------------------------------------------------------------------------
# Rich prediction object
# ---------------------------------------------------------------------------


@dataclass
class DamageDetectionPrediction:
    """Post-processed output of one damage detection inference call."""

    result: DetectionResult
    class_names: list[str]
    original_size: tuple[int, int]
    model_size: tuple[int, int]

    @property
    def num_detections(self) -> int:
        return len(self.result.instances)

    def visualize(self, image: np.ndarray) -> np.ndarray:
        """Draw detection boxes onto *image* and return the annotated array."""
        return draw_detections(
            image,
            self.result.instances,
            class_names=self.class_names,
            model_size=self.model_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "num_detections": self.num_detections,
            "detections": [
                {
                    "label": inst.label,
                    "label_name": (
                        self.class_names[inst.label]
                        if inst.label < len(self.class_names)
                        else "unknown"
                    ),
                    "score": inst.score,
                    "box": inst.box,
                }
                for inst in self.result.instances
            ],
        }
