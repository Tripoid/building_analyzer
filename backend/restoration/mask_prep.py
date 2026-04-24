"""
Merge all defect masks into a single inpaint mask for the restoration API.

Workflow:
    1. Union every defect class mask.
    2. Dilate by `dilate_kernel` to cover feathered edges the segmentation
       tends to miss (cracks often bleed slightly past the mask boundary).
    3. Feather via Gaussian blur so the inpainted seam blends naturally.

Returns an HxW uint8 array where 255 = "repaint this pixel", 0 = "keep".
"""

from __future__ import annotations

from functools import reduce

import cv2
import numpy as np


def prepare_restoration_mask(
    defect_masks: dict[str, np.ndarray],
    dilate_kernel: int = 15,
    dilate_iters: int = 2,
    feather_sigma: float = 8.0,
) -> np.ndarray:
    if not defect_masks:
        return np.zeros((1, 1), dtype=np.uint8)

    boolean_masks = [m.astype(bool) for m in defect_masks.values() if m is not None]
    if not boolean_masks:
        return np.zeros_like(next(iter(defect_masks.values())), dtype=np.uint8)

    union = reduce(np.logical_or, boolean_masks)
    union_u8 = union.astype(np.uint8) * 255

    kernel_size = max(3, dilate_kernel | 1)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(union_u8, kernel, iterations=dilate_iters)

    if feather_sigma > 0:
        dilated = cv2.GaussianBlur(dilated, (0, 0), feather_sigma)
        # Preserve high-confidence centre at 255
        dilated = np.clip(dilated * 1.2, 0, 255).astype(np.uint8)

    return dilated
