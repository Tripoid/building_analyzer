"""Utility functions for the facade segmentation module."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ml.facade_segmentation.dataset import FACADE_CLASS_NAMES

# ---------------------------------------------------------------------------
# Colour palette (one distinct RGB colour per class)
# ---------------------------------------------------------------------------

#: Default colour palette — one ``(R, G, B)`` tuple per class index.
FACADE_COLOUR_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0  background  — black
    (128, 0, 0),      # 1  wall        — dark red
    (0, 128, 0),      # 2  window      — green
    (128, 128, 0),    # 3  door        — olive
    (0, 0, 128),      # 4  balcony     — dark blue
    (128, 0, 128),    # 5  cornice     — purple
    (255, 0, 0),      # 6  damaged     — bright red
]

# Extend palette to 256 entries so we can always index by class index safely
_EXTENDED_PALETTE: list[tuple[int, int, int]] = FACADE_COLOUR_PALETTE + [
    (200, 200, 200)
] * (256 - len(FACADE_COLOUR_PALETTE))


def colorize_mask(
    mask: np.ndarray,
    palette: list[tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Convert an integer segmentation mask to a coloured RGB PIL image.

    Args:
        mask: 2-D integer array of shape (H, W) with class indices.
        palette: Optional list of ``(R, G, B)`` tuples.  Defaults to
            :data:`FACADE_COLOUR_PALETTE`.

    Returns:
        RGB PIL image of shape (H, W, 3).
    """
    pal = palette or _EXTENDED_PALETTE
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(pal):
        rgb[mask == cls_idx] = color
    return Image.fromarray(rgb, mode="RGB")


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    palette: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Blend a coloured segmentation mask onto an RGB image.

    Args:
        image: RGB image array of shape (H, W, 3), dtype uint8.
        mask: Integer mask of shape (H_m, W_m).  Resized to *image* size
            if dimensions differ.
        alpha: Blend weight for the overlay (0 = image only, 1 = mask only).
        palette: Optional colour palette.

    Returns:
        Blended RGB uint8 array of the same shape as *image*.
    """
    h, w = image.shape[:2]
    if mask.shape != (h, w):
        from PIL import Image as _Image
        mask_pil = _Image.fromarray(mask.astype(np.uint8)).resize(
            (w, h), resample=_Image.NEAREST
        )
        mask = np.array(mask_pil)

    colored = np.array(colorize_mask(mask, palette=palette), dtype=np.float32)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * colored
    return blended.clip(0, 255).astype(np.uint8)


def extract_class_masks(
    mask: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return a binary mask for each class present in *mask*.

    Args:
        mask: Integer class-index array of shape (H, W).
        class_names: Names to use as keys.  Defaults to
            :data:`~ml.facade_segmentation.dataset.FACADE_CLASS_NAMES`.

    Returns:
        Dict mapping class name → boolean (H, W) array.
    """
    names = class_names or FACADE_CLASS_NAMES
    return {
        name: (mask == i).astype(bool)
        for i, name in enumerate(names)
        if (mask == i).any()
    }


def compute_class_statistics(
    mask: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute area statistics for each class in *mask*.

    Args:
        mask: Integer class-index array of shape (H, W).
        class_names: Names to use as keys.

    Returns:
        Dict mapping class name → stats dict with keys
        ``"pixel_count"`` and ``"area_fraction"``.
    """
    names = class_names or FACADE_CLASS_NAMES
    total = mask.size
    stats: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        count = int((mask == i).sum())
        stats[name] = {
            "pixel_count": float(count),
            "area_fraction": count / total,
        }
    return stats
