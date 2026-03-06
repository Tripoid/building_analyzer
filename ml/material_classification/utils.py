"""Utility functions for the material classification module."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from PIL import Image


def extract_region_crops(
    image: np.ndarray,
    boxes: list[list[float]],
    target_size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Extract and optionally resize image crops for each bounding box.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        boxes: List of ``[x1, y1, x2, y2]`` in pixel coordinates.
        target_size: If provided, resize each crop to ``(height, width)``.

    Returns:
        List of RGB uint8 arrays, one per box.
    """
    h, w = image.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            # Degenerate box — return a blank crop
            crop = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            crop = image[y1:y2, x1:x2]

        if target_size is not None:
            th, tw = target_size
            pil_crop = Image.fromarray(crop).resize((tw, th), Image.BILINEAR)
            crop = np.array(pil_crop, dtype=np.uint8)

        crops.append(crop)
    return crops


def aggregate_region_materials(
    region_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-region material predictions into a summary.

    Args:
        region_results: List of result dicts from
            :meth:`~ml.material_classification.inference.MaterialInferencer.predict_regions`.

    Returns:
        Dict with keys:
        - ``"dominant_material"`` — most common predicted material name.
        - ``"material_counts"`` — Counter of label_name → count.
        - ``"average_scores"`` — dict of label_name → mean confidence score.
    """
    if not region_results:
        return {
            "dominant_material": None,
            "material_counts": {},
            "average_scores": {},
        }

    label_names = [r["label_name"] for r in region_results]
    counts: Counter = Counter(label_names)
    dominant = counts.most_common(1)[0][0]

    # Average score per class from ``scores`` dicts
    score_accum: dict[str, list[float]] = {}
    for r in region_results:
        for cls_name, score in r.get("scores", {}).items():
            score_accum.setdefault(cls_name, []).append(score)
    avg_scores = {cls: float(np.mean(vals)) for cls, vals in score_accum.items()}

    return {
        "dominant_material": dominant,
        "material_counts": dict(counts),
        "average_scores": avg_scores,
    }


def gradcam_heatmap(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    image_tensor: "torch.Tensor",  # type: ignore[name-defined]
    target_class: int,
    target_layer: "torch.nn.Module",  # type: ignore[name-defined]
) -> np.ndarray:
    """Compute a GradCAM heatmap for *image_tensor* with respect to *target_class*.

    Args:
        model: A classification model in eval mode.
        image_tensor: Float tensor of shape (1, 3, H, W).
        target_class: Class index to explain.
        target_layer: Convolutional layer from which to extract gradients.

    Returns:
        Normalised heatmap as a float32 array of shape (H_feat, W_feat)
        with values in [0, 1].
    """
    import torch

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(module, inp, out):  # noqa: ANN001, ANN201
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):  # noqa: ANN001, ANN201
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(image_tensor)
    logits[0, target_class].backward()

    fwd_handle.remove()
    bwd_handle.remove()

    if not activations or not gradients:
        return np.zeros((1, 1), dtype=np.float32)

    act = activations[0][0]  # (C, H, W)
    grad = gradients[0][0]  # (C, H, W)

    weights = grad.mean(dim=(1, 2))  # (C,)
    cam = (weights[:, None, None] * act).sum(dim=0)  # (H, W)
    cam = torch.relu(cam).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)
