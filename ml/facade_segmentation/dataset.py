"""Facade segmentation dataset.

Expects the dataset to be organised as follows::

    root/
        images/
            {split}/
                image_001.jpg
                ...
        masks/
            {split}/
                image_001.png   ← single-channel PNG, pixel = class index
                ...

Class labels
------------
0  background
1  wall
2  window
3  door
4  balcony
5  cornice
6  damaged
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image

from ml.common.base_dataset import BaseDataset
from ml.common.transforms import get_segmentation_transforms

# Canonical label space for facade segmentation
FACADE_CLASS_NAMES: list[str] = [
    "background",
    "wall",
    "window",
    "door",
    "balcony",
    "cornice",
    "damaged",
]


class FacadeSegmentationDataset(BaseDataset):
    """PyTorch dataset for facade semantic segmentation.

    Each sample is a ``(image_tensor, mask_tensor)`` tuple where:

    * ``image_tensor`` has shape ``(3, H, W)`` and dtype ``float32``.
    * ``mask_tensor`` has shape ``(H, W)`` and dtype ``int64`` with values
      in ``[0, num_classes - 1]``.

    Args:
        root: Dataset root directory (must contain ``images/`` and ``masks/``
            sub-directories with the matching split sub-directory inside each).
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        image_size: Target ``(height, width)`` passed to the transform pipeline.
        transform: Optional custom albumentations ``Compose``.  When ``None``
            a default pipeline is built from
            :func:`~ml.common.transforms.get_segmentation_transforms`.
        ignore_index: Pixel value in the mask PNG that marks ignored regions.
    """

    CLASS_NAMES: list[str] = FACADE_CLASS_NAMES

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (512, 512),
        transform: A.Compose | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__(root=root, split=split)
        self.ignore_index = ignore_index

        self._image_dir = self.root / "images" / split
        self._mask_dir = self.root / "masks" / split

        self._samples: list[tuple[Path, Path]] = self._discover_samples()

        self._transform = transform or get_segmentation_transforms(
            image_size=image_size,
            is_train=(split == "train"),
        )

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def class_names(self) -> list[str]:
        return self.CLASS_NAMES

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self._samples[index]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)

        augmented: dict[str, Any] = self._transform(image=image, mask=mask)
        image_tensor: torch.Tensor = augmented["image"]
        mask_tensor: torch.Tensor = augmented["mask"].long()

        return image_tensor, mask_tensor

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _discover_samples(self) -> list[tuple[Path, Path]]:
        """Pair each image file with its corresponding mask file."""
        if not self._image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self._image_dir}. "
                "Ensure the dataset is organised as root/images/{split}/ and "
                "root/masks/{split}/."
            )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        samples: list[tuple[Path, Path]] = []

        for img_path in sorted(self._image_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue
            # Masks are stored as PNGs with the same stem
            mask_path = self._mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image '{img_path}'. "
                    f"Expected: '{mask_path}'."
                )
            samples.append((img_path, mask_path))

        if not samples:
            raise RuntimeError(
                f"No valid image–mask pairs found under '{self._image_dir}'. "
                "Check that the directory is not empty."
            )
        return samples
