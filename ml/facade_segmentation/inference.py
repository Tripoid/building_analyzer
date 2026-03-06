"""Facade segmentation inferencer.

Provides single-image and batch inference with configurable pre/post-processing,
including:
- automatic device selection
- image normalisation matching training transforms
- optional test-time augmentation (TTA) via horizontal flip
- structured result objects
"""

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

from ml.common.base_model import BaseSegmentationModel, SegmentationResult
from ml.common.registry import ModelRegistry
from ml.facade_segmentation.dataset import FACADE_CLASS_NAMES
from ml.facade_segmentation.utils import colorize_mask, overlay_mask_on_image

logger = logging.getLogger(__name__)


@dataclass
class SegmentationInferencerConfig:
    """Configuration for the segmentation inferencer."""

    model_name: str = "deeplabv3plus"
    model_weights: str | None = None
    device: str = "auto"
    image_size: tuple[int, int] = (512, 512)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    tta: bool = False  # test-time augmentation (horizontal flip)
    score_threshold: float = 0.5  # minimum per-pixel probability kept in output
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class SegmentationInferencer:
    """Runs facade segmentation inference on arbitrary images.

    Handles all pre-processing (resize, normalise, tensorise) and
    post-processing (argmax, optional TTA averaging, colourisation) so that
    callers can pass raw PIL images or file paths.

    Example::

        inferencer = SegmentationInferencer(model)
        result = inferencer.predict_from_path("photo.jpg")
        colored = result.colored_mask  # RGB PIL Image

    Args:
        model: A loaded :class:`~ml.common.base_model.BaseSegmentationModel`.
        config: Inferencer configuration.
    """

    def __init__(
        self,
        model: BaseSegmentationModel,
        config: SegmentationInferencerConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = config or SegmentationInferencerConfig()
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
        self._flip_transform: A.Compose = A.Compose(
            [
                A.Resize(h, w),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=self.cfg.mean, std=self.cfg.std),
                ToTensorV2(),
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_path(self, path: str | Path) -> "SegmentationPrediction":
        """Load an image from *path* and run inference.

        Args:
            path: Path to an image file (any format PIL can open).

        Returns:
            A :class:`SegmentationPrediction` with the segmentation mask and
            metadata.
        """
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return self.predict_from_array(image)

    def predict_from_array(self, image: np.ndarray) -> "SegmentationPrediction":
        """Run inference on a single HWC uint8 NumPy image.

        Args:
            image: RGB image array of shape (H, W, 3) with values in [0, 255].

        Returns:
            A :class:`SegmentationPrediction`.
        """
        original_h, original_w = image.shape[:2]
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            logits = self.model(tensor)  # (1, C, H, W)
            if self.cfg.tta:
                flip_tensor = self._preprocess_flip(image).unsqueeze(0).to(self.device)
                flip_logits = self.model(flip_tensor)
                # Flip back and average
                flip_logits = torch.flip(flip_logits, dims=[-1])
                logits = (logits + flip_logits) / 2.0

        probs = torch.softmax(logits[0], dim=0)  # (C, H, W)
        mask = torch.argmax(probs, dim=0)  # (H, W)

        return SegmentationPrediction(
            mask=mask.cpu(),
            probabilities=probs.cpu(),
            class_names=self.model.class_names,
            original_size=(original_h, original_w),
        )

    def predict_batch(self, images: list[np.ndarray]) -> list["SegmentationPrediction"]:
        """Run inference on a batch of HWC uint8 NumPy arrays.

        Args:
            images: List of RGB images, each of shape (H, W, 3).

        Returns:
            List of :class:`SegmentationPrediction` objects.
        """
        original_sizes = [img.shape[:2] for img in images]
        tensors = torch.stack(
            [self._preprocess(img) for img in images], dim=0
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(tensors)  # (B, C, H, W)

        results = []
        for i, (oh, ow) in enumerate(original_sizes):
            probs = torch.softmax(logits[i], dim=0).cpu()
            mask = torch.argmax(probs, dim=0)
            results.append(
                SegmentationPrediction(
                    mask=mask,
                    probabilities=probs,
                    class_names=self.model.class_names,
                    original_size=(oh, ow),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        return self._transform(image=image)["image"]  # type: ignore[return-value]

    def _preprocess_flip(self, image: np.ndarray) -> torch.Tensor:
        return self._flip_transform(image=image)["image"]  # type: ignore[return-value]

    @classmethod
    def from_config(
        cls,
        config: SegmentationInferencerConfig,
        **model_kwargs: Any,
    ) -> "SegmentationInferencer":
        """Instantiate from *config*, building the model from the registry.

        Args:
            config: Inferencer configuration (must have *model_name* set).
            **model_kwargs: Extra kwargs forwarded to the model constructor.

        Returns:
            A fully configured :class:`SegmentationInferencer`.
        """
        model: BaseSegmentationModel = ModelRegistry.build(
            config.model_name, namespace="segmentation", **model_kwargs
        )
        if config.model_weights:
            model.load(config.model_weights, map_location=config.resolve_device())
        return cls(model=model, config=config)


# ---------------------------------------------------------------------------
# Rich prediction object
# ---------------------------------------------------------------------------


@dataclass
class SegmentationPrediction:
    """Post-processed output of one segmentation inference call.

    Attributes:
        mask: Integer class-index tensor of shape (H_model, W_model).
        probabilities: Float probability tensor of shape (C, H_model, W_model).
        class_names: Ordered list of class names.
        original_size: ``(height, width)`` of the original input image before
            resizing, useful for restoring coordinates.
    """

    mask: torch.Tensor
    probabilities: torch.Tensor
    class_names: list[str]
    original_size: tuple[int, int]

    @property
    def colored_mask(self) -> Image.Image:
        """Return an RGB PIL image with a distinct colour per class."""
        return colorize_mask(self.mask.numpy())

    def overlay_on(self, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Return an overlay of the coloured mask on *image*.

        Args:
            image: RGB uint8 array of any size.
            alpha: Blending factor for the mask overlay.

        Returns:
            RGB uint8 array of the same size as *image*.
        """
        return overlay_mask_on_image(image, self.mask.numpy(), alpha=alpha)

    def class_area_fractions(self) -> dict[str, float]:
        """Return the fraction of pixels assigned to each class."""
        total = self.mask.numel()
        return {
            name: float((self.mask == i).sum().item()) / total
            for i, name in enumerate(self.class_names)
        }
