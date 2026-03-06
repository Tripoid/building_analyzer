"""Material classification dataset.

Expects an ImageFolder-style layout, i.e. one sub-directory per class::

    root/
        train/
            concrete/
                img_001.jpg
                ...
            brick/
                ...
        val/
            concrete/
                ...
            brick/
                ...
        test/
            ...

Alternatively, a CSV manifest can be provided::

    root/
        manifest.csv    ← columns: ``path,label``
        images/
            img_001.jpg
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image

from ml.common.base_dataset import BaseDataset
from ml.common.transforms import get_classification_transforms

MATERIAL_CLASS_NAMES: list[str] = [
    "concrete",
    "brick",
    "glass",
    "wood",
    "metal",
    "stone",
]


class MaterialDataset(BaseDataset):
    """PyTorch dataset for building material classification.

    Supports two data layouts:

    1. **ImageFolder** — sub-directories named after classes (default).
    2. **CSV manifest** — a ``manifest.csv`` file at *root* with columns
       ``path`` (relative to *root*) and ``label`` (integer class index).

    Each ``__getitem__`` returns ``(image_tensor, label)`` where
    ``image_tensor`` has shape ``(3, H, W)`` and ``label`` is an int64 scalar.

    Args:
        root: Dataset root directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        image_size: Target ``(height, width)`` for the transform pipeline.
        transform: Optional custom albumentations ``Compose`` pipeline.
        use_csv: If ``True``, load from ``root/manifest.csv`` instead of
            the ImageFolder layout.
    """

    CLASS_NAMES: list[str] = MATERIAL_CLASS_NAMES

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (224, 224),
        transform: A.Compose | None = None,
        use_csv: bool = False,
    ) -> None:
        super().__init__(root=root, split=split)

        self._transform = transform or get_classification_transforms(
            image_size=image_size,
            is_train=(split == "train"),
        )

        if use_csv:
            self._samples = self._load_from_csv()
        else:
            self._samples = self._load_imagefolder()

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label = self._samples[index]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        aug: dict[str, Any] = self._transform(image=image)
        image_tensor: torch.Tensor = aug["image"]
        return image_tensor, label

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_imagefolder(self) -> list[tuple[Path, int]]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                "Expected root/{split}/{class_name}/ layout."
            )
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        name_to_idx = {name: i for i, name in enumerate(self.CLASS_NAMES)}

        samples: list[tuple[Path, int]] = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in name_to_idx:
                continue  # skip unknown classes gracefully
            label = name_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    samples.append((img_path, label))

        if not samples:
            raise RuntimeError(
                f"No samples found in '{split_dir}'. "
                "Ensure sub-directories match class names: "
                f"{self.CLASS_NAMES}."
            )
        return samples

    def _load_from_csv(self) -> list[tuple[Path, int]]:
        csv_path = self.root / "manifest.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

        samples: list[tuple[Path, int]] = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", self.split) != self.split:
                    continue
                img_path = self.root / row["path"]
                label = int(row["label"])
                samples.append((img_path, label))

        if not samples:
            raise RuntimeError(f"No samples found in '{csv_path}' for split '{self.split}'.")
        return samples
