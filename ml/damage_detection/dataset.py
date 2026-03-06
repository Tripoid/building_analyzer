"""Damage detection dataset.

Loads images with COCO-style JSON annotations describing damage instances.

Expected dataset layout::

    root/
        images/
            {split}/
                img_001.jpg
                ...
        annotations/
            {split}.json    ← COCO-format annotation file

COCO annotation format (minimal subset used here)::

    {
        "images":      [{"id": 1, "file_name": "img_001.jpg", "height": H, "width": W}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                          "bbox": [x, y, w, h], "segmentation": [...], "area": A}],
        "categories":  [{"id": 1, "name": "crack"}, ...]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image

from ml.common.base_dataset import BaseDataset
from ml.common.transforms import get_detection_transforms

DAMAGE_CLASS_NAMES: list[str] = [
    "crack",
    "spalling",
    "corrosion",
    "delamination",
    "efflorescence",
]


class DamageDetectionDataset(BaseDataset):
    """PyTorch dataset for building damage detection.

    Each ``__getitem__`` call returns a tuple ``(image_tensor, target)`` where:

    * ``image_tensor`` — float32 tensor of shape ``(3, H, W)``.
    * ``target`` — dict with:

      - ``"boxes"`` — float32 tensor (N, 4) in ``[x1, y1, x2, y2]`` format.
      - ``"labels"`` — int64 tensor (N,) with 1-based class indices.
      - ``"image_id"`` — int64 scalar.
      - ``"area"`` — float32 tensor (N,) with annotation areas.
      - ``"iscrowd"`` — int64 tensor (N,) crowd flags.

    Args:
        root: Dataset root directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        image_size: Target ``(height, width)`` for the transform pipeline.
        transform: Optional custom albumentations ``Compose`` pipeline.
        min_box_area: Minimum pixel area for a bounding box to be kept.
    """

    CLASS_NAMES: list[str] = DAMAGE_CLASS_NAMES

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (640, 640),
        transform: A.Compose | None = None,
        min_box_area: float = 1.0,
    ) -> None:
        super().__init__(root=root, split=split)
        self.min_box_area = min_box_area

        self._image_dir = self.root / "images" / split
        ann_path = self.root / "annotations" / f"{split}.json"

        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}. "
                "Provide a COCO-format JSON at root/annotations/{split}.json."
            )

        with ann_path.open() as f:
            coco = json.load(f)

        self._build_index(coco)

        self._transform = transform or get_detection_transforms(
            image_size=image_size,
            is_train=(split == "train"),
            bbox_format="pascal_voc",
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
        return len(self._image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img_id = self._image_ids[index]
        img_info = self._images[img_id]
        img_path = self._image_dir / img_info["file_name"]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        annotations = self._ann_by_image.get(img_id, [])

        # Parse boxes and labels
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for ann in annotations:
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            area = bw * bh
            if area < self.min_box_area:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))
            areas.append(float(area))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        # Apply transform (handles box coordinate updates)
        if boxes:
            aug = self._transform(
                image=image,
                bboxes=boxes,
                class_labels=labels,
            )
            image_tensor: torch.Tensor = aug["image"]
            boxes = [list(b) for b in aug["bboxes"]]
            labels = list(aug["class_labels"])
        else:
            aug = self._transform(image=image, bboxes=[], class_labels=[])
            image_tensor = aug["image"]

        target: dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_index(self, coco: dict) -> None:
        self._images: dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
        self._image_ids: list[int] = list(self._images.keys())

        self._ann_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            self._ann_by_image.setdefault(img_id, []).append(ann)

        # Remap category ids to be 1-based and contiguous
        categories = coco.get("categories", [])
        if categories:
            self._cat_id_to_label: dict[int, int] = {
                cat["id"]: i + 1 for i, cat in enumerate(categories)
            }
        else:
            self._cat_id_to_label = {}
