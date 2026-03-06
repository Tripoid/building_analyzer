"""Utility functions for the damage detection module."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops

from ml.common.base_model import DetectionInstance

# ---------------------------------------------------------------------------
# Colours per damage category
# ---------------------------------------------------------------------------

DAMAGE_COLOURS: list[tuple[int, int, int]] = [
    (255, 0, 0),     # crack         — red
    (255, 165, 0),   # spalling      — orange
    (0, 0, 255),     # corrosion     — blue
    (128, 0, 128),   # delamination  — purple
    (0, 255, 0),     # efflorescence — green
]


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def compute_box_iou_matrix(boxes: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU matrix for a set of boxes.

    Args:
        boxes: Float tensor of shape (N, 4) in ``[x1, y1, x2, y2]`` format.

    Returns:
        Symmetric float tensor of shape (N, N).
    """
    if boxes.numel() == 0:
        return torch.zeros(0, 0)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    inter_x1 = torch.max(x1[:, None], x1[None, :])
    inter_y1 = torch.max(y1[:, None], y1[None, :])
    inter_x2 = torch.min(x2[:, None], x2[None, :])
    inter_y2 = torch.min(y2[:, None], y2[None, :])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = areas[:, None] + areas[None, :] - inter
    return inter / union.clamp(min=1e-6)


def apply_nms(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply class-aware Non-Maximum Suppression using torchvision.

    Args:
        boxes: Float tensor (N, 4).
        labels: Int tensor (N,).
        scores: Float tensor (N,).
        score_threshold: Minimum score to keep.
        iou_threshold: NMS IoU threshold.
        max_detections: Maximum number of detections to return.

    Returns:
        Tuple of ``(boxes, labels, scores)`` after NMS, each a CPU tensor.
    """
    # Score filtering
    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask]
    labels = labels[keep_mask]
    scores = scores[keep_mask]

    if boxes.numel() == 0:
        return boxes, labels, scores

    # Class-aware NMS via torchvision (fast C++ implementation)
    # Offset boxes by class to achieve per-class NMS behaviour
    max_coord = boxes.max() + 1
    offsets = labels.float() * (max_coord + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep_indices = tv_ops.nms(boxes_for_nms, scores, iou_threshold)

    # Sort kept indices by score descending
    sorted_order = scores[keep_indices].argsort(descending=True)
    keep_indices = keep_indices[sorted_order]

    # Limit
    if keep_indices.shape[0] > max_detections:
        keep_indices = keep_indices[:max_detections]

    return boxes[keep_indices], labels[keep_indices], scores[keep_indices]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_detections(
    image: np.ndarray,
    instances: list[DetectionInstance],
    class_names: list[str],
    model_size: tuple[int, int] | None = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on *image*.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        instances: List of :class:`~ml.common.base_model.DetectionInstance`.
        class_names: List of class name strings for label rendering.
        model_size: ``(H_model, W_model)`` used to scale box coordinates back
            to the original image size.  If ``None``, boxes are assumed to
            already be in image pixel coordinates.
        line_thickness: Width of the drawn rectangle border.

    Returns:
        RGB uint8 array with bounding boxes drawn.
    """
    result = image.copy()
    img_h, img_w = image.shape[:2]

    for inst in instances:
        x1, y1, x2, y2 = inst.box
        if model_size is not None:
            mh, mw = model_size
            x1 = x1 / mw * img_w
            y1 = y1 / mh * img_h
            x2 = x2 / mw * img_w
            y2 = y2 / mh * img_h

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = DAMAGE_COLOURS[inst.label % len(DAMAGE_COLOURS)]

        # Draw rectangle
        for t in range(line_thickness):
            result[
                max(0, y1 + t): max(0, y2 - t),
                max(0, x1 + t): max(0, x1 + t + 1),
            ] = color
            result[
                max(0, y1 + t): max(0, y2 - t),
                max(0, x2 - t): max(0, x2 - t + 1),
            ] = color
            result[
                max(0, y1 + t): max(0, y1 + t + 1),
                max(0, x1 + t): max(0, x2 - t),
            ] = color
            result[
                max(0, y2 - t): max(0, y2 - t + 1),
                max(0, x1 + t): max(0, x2 - t),
            ] = color

    return result


def scale_boxes_to_original(
    boxes: list[list[float]],
    model_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list[list[float]]:
    """Rescale boxes from model-input coordinates to original image coordinates.

    Args:
        boxes: List of ``[x1, y1, x2, y2]`` in model space.
        model_size: ``(H_model, W_model)``.
        original_size: ``(H_orig, W_orig)``.

    Returns:
        Rescaled boxes in original image coordinates.
    """
    mh, mw = model_size
    oh, ow = original_size
    scale_x = ow / mw
    scale_y = oh / mh
    return [
        [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
        for b in boxes
    ]
