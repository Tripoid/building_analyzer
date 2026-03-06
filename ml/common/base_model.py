"""Abstract base classes for all ML models in the building analyzer system.

Every concrete model must subclass one of the three bases and implement the
abstract methods, guaranteeing a uniform interface for training, inference,
and the pipeline orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """Per-image output of a segmentation model."""

    mask: torch.Tensor  # (H, W) int64 class indices
    logits: torch.Tensor  # (C, H, W) raw logits before softmax
    probabilities: torch.Tensor  # (C, H, W) softmax probabilities


@dataclass
class DetectionInstance:
    """A single detected instance (bounding box + optional mask)."""

    label: int
    score: float
    box: list[float]  # [x1, y1, x2, y2] in pixel coordinates
    mask: torch.Tensor | None = None  # (H, W) binary mask, optional


@dataclass
class DetectionResult:
    """Per-image output of a detection model."""

    instances: list[DetectionInstance] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Per-crop output of a classification model."""

    label: int
    label_name: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)  # class_name → prob


# ---------------------------------------------------------------------------
# Abstract base models
# ---------------------------------------------------------------------------


class BaseSegmentationModel(nn.Module, ABC):
    """Interface for semantic segmentation models.

    Subclasses must implement :meth:`forward` (raw logits) and :meth:`predict`
    (post-processed :class:`SegmentationResult`).  The number of classes and the
    class-name mapping are exposed as public attributes so that downstream code
    can query them without knowing the concrete model type.
    """

    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        super().__init__()
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names length {len(class_names)} != num_classes {num_classes}"
            )
        self.num_classes = num_classes
        self.class_names = class_names

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, C, H, W).

        Args:
            images: Float tensor of shape (B, 3, H, W).

        Returns:
            Logit tensor of shape (B, C, H, W).
        """

    def predict(self, images: torch.Tensor) -> list[SegmentationResult]:
        """Run forward pass and post-process into per-image results.

        Args:
            images: Float tensor of shape (B, 3, H, W).

        Returns:
            List of :class:`SegmentationResult`, one per image.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)  # (B, C, H, W)
            probs = torch.softmax(logits, dim=1)
            masks = torch.argmax(probs, dim=1)  # (B, H, W)

        results = []
        for i in range(images.shape[0]):
            results.append(
                SegmentationResult(
                    mask=masks[i],
                    logits=logits[i],
                    probabilities=probs[i],
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        """Save model weights to *path*."""
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location: Any = "cpu") -> None:
        """Load model weights from *path*."""
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)


class BaseDetectionModel(nn.Module, ABC):
    """Interface for object detection / instance segmentation models.

    Implementations wrap any detector (anchor-based, anchor-free, transformer,
    …) and expose a uniform :meth:`predict` API that returns
    :class:`DetectionResult` objects.
    """

    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        super().__init__()
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names length {len(class_names)} != num_classes {num_classes}"
            )
        self.num_classes = num_classes
        self.class_names = class_names

    @abstractmethod
    def forward(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return raw detector outputs.

        Args:
            images: Float tensor of shape (B, 3, H, W).

        Returns:
            List of per-image dicts with at least the keys
            ``"boxes"`` (N,4), ``"labels"`` (N,), ``"scores"`` (N,).
            An optional ``"masks"`` (N, H, W) key may also be present.
        """

    def predict(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.5,
    ) -> list[DetectionResult]:
        """Run forward pass and filter by *score_threshold*.

        Args:
            images: Float tensor of shape (B, 3, H, W).
            score_threshold: Minimum confidence to keep a detection.

        Returns:
            List of :class:`DetectionResult`, one per image.
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(images)

        results = []
        for preds in raw:
            boxes = preds.get("boxes", torch.zeros(0, 4))
            labels = preds.get("labels", torch.zeros(0, dtype=torch.long))
            scores = preds.get("scores", torch.zeros(0))
            masks_tensor = preds.get("masks")  # may be None

            keep = scores >= score_threshold
            instances = []
            for idx in keep.nonzero(as_tuple=False).squeeze(1):
                i = idx.item()
                mask = masks_tensor[i] if masks_tensor is not None else None
                instances.append(
                    DetectionInstance(
                        label=int(labels[i].item()),
                        score=float(scores[i].item()),
                        box=boxes[i].tolist(),
                        mask=mask,
                    )
                )
            results.append(DetectionResult(instances=instances))
        return results

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location: Any = "cpu") -> None:
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)


class BaseClassificationModel(nn.Module, ABC):
    """Interface for image / region classification models.

    Accepts arbitrary image crops and returns per-class probabilities.
    """

    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        super().__init__()
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names length {len(class_names)} != num_classes {num_classes}"
            )
        self.num_classes = num_classes
        self.class_names = class_names

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, num_classes).

        Args:
            images: Float tensor of shape (B, 3, H, W).

        Returns:
            Logit tensor of shape (B, num_classes).
        """

    def predict(self, images: torch.Tensor) -> list[ClassificationResult]:
        """Run forward pass and return per-image classification results.

        Args:
            images: Float tensor of shape (B, 3, H, W).

        Returns:
            List of :class:`ClassificationResult`, one per image.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)  # (B, C)
            probs = torch.softmax(logits, dim=1)  # (B, C)
            top_labels = torch.argmax(probs, dim=1)  # (B,)

        results = []
        for i in range(images.shape[0]):
            label = int(top_labels[i].item())
            score = float(probs[i, label].item())
            scores_dict = {
                self.class_names[c]: float(probs[i, c].item())
                for c in range(self.num_classes)
            }
            results.append(
                ClassificationResult(
                    label=label,
                    label_name=self.class_names[label],
                    score=score,
                    scores=scores_dict,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location: Any = "cpu") -> None:
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)
