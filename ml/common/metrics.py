"""Metric helpers used by trainers and evaluators across all ML modules.

All functions operate on plain Python / NumPy / PyTorch primitives so they can
be called from training loops, evaluation scripts, and notebooks without extra
dependencies.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


def compute_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Compute per-class and mean Intersection-over-Union (mIoU).

    Args:
        pred_mask: Predicted class indices, shape (H, W) or (B, H, W).
        gt_mask: Ground-truth class indices, same shape as *pred_mask*.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude from evaluation.

    Returns:
        Dictionary with keys ``"per_class_iou"`` (list[float]) and
        ``"mean_iou"`` (float).
    """
    pred_mask = pred_mask.view(-1).long()
    gt_mask = gt_mask.view(-1).long()

    valid = gt_mask != ignore_index
    pred_mask = pred_mask[valid]
    gt_mask = gt_mask[valid]

    per_class_iou: list[float] = []
    for cls in range(num_classes):
        pred_cls = pred_mask == cls
        gt_cls = gt_mask == cls
        intersection = (pred_cls & gt_cls).sum().item()
        union = (pred_cls | gt_cls).sum().item()
        if union == 0:
            # Class absent from both prediction and GT — skip
            continue
        per_class_iou.append(intersection / union)

    mean_iou = float(np.mean(per_class_iou)) if per_class_iou else 0.0
    return {"per_class_iou": per_class_iou, "mean_iou": mean_iou}


def compute_pixel_accuracy(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """Compute overall pixel accuracy.

    Args:
        pred_mask: Predicted class indices, shape (H, W) or (B, H, W).
        gt_mask: Ground-truth class indices, same shape.
        ignore_index: Class index to exclude.

    Returns:
        Pixel accuracy as a float in [0, 1].
    """
    pred_mask = pred_mask.view(-1).long()
    gt_mask = gt_mask.view(-1).long()

    valid = gt_mask != ignore_index
    pred_mask = pred_mask[valid]
    gt_mask = gt_mask[valid]

    if valid.sum() == 0:
        return 0.0
    return float((pred_mask == gt_mask).float().mean().item())


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return IoU between two axis-aligned boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute mean Average Precision (mAP) at a given IoU threshold.

    Args:
        predictions: List of per-image dicts, each with keys:
            ``"boxes"`` (N,4), ``"labels"`` (N,), ``"scores"`` (N,).
        ground_truths: List of per-image dicts, each with keys:
            ``"boxes"`` (M,4), ``"labels"`` (M,).
        iou_threshold: IoU threshold for a positive match.
        num_classes: If not provided, inferred from the annotations.

    Returns:
        Dictionary with keys ``"per_class_ap"`` (dict[int, float]) and
        ``"map"`` (float).
    """
    # Gather all class ids
    all_labels: set[int] = set()
    for gt in ground_truths:
        all_labels.update(int(l) for l in gt.get("labels", []))
    if num_classes is not None:
        all_labels.update(range(num_classes))

    per_class_ap: dict[int, float] = {}

    for cls in all_labels:
        # Collect all predictions for this class, sorted by score descending
        all_preds: list[tuple[float, int, int]] = []  # (score, image_idx, box_idx)
        for img_idx, pred in enumerate(predictions):
            boxes = pred.get("boxes", [])
            labels = pred.get("labels", [])
            scores = pred.get("scores", [])
            for j, (label, score) in enumerate(zip(labels, scores)):
                if int(label) == cls:
                    all_preds.append((float(score), img_idx, j))
        all_preds.sort(reverse=True, key=lambda x: x[0])

        # Count ground-truth instances
        gt_counts = [
            sum(1 for l in gt.get("labels", []) if int(l) == cls)
            for gt in ground_truths
        ]
        total_gt = sum(gt_counts)
        if total_gt == 0:
            continue

        matched: list[list[bool]] = [
            [False] * gt_counts[i] for i in range(len(ground_truths))
        ]

        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        for rank, (score, img_idx, box_idx) in enumerate(all_preds):
            pred_box = np.array(predictions[img_idx]["boxes"][box_idx])
            gt_boxes = [
                np.array(b)
                for b, l in zip(
                    ground_truths[img_idx].get("boxes", []),
                    ground_truths[img_idx].get("labels", []),
                )
                if int(l) == cls
            ]
            best_iou, best_gt_idx = 0.0, -1
            for gi, gt_box in enumerate(gt_boxes):
                iou = _box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gi

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_cls_idx = best_gt_idx
                if not matched[img_idx][gt_cls_idx]:
                    tp[rank] = 1
                    matched[img_idx][gt_cls_idx] = True
                else:
                    fp[rank] = 1
            else:
                fp[rank] = 1

        cumtp = np.cumsum(tp)
        cumfp = np.cumsum(fp)
        recall = cumtp / total_gt
        precision = cumtp / (cumtp + cumfp + 1e-9)

        # Compute area under PR curve (11-point interpolation)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p_at_r = precision[recall >= t].max() if (recall >= t).any() else 0.0
            ap += p_at_r / 11
        per_class_ap[cls] = float(ap)

    map_score = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {"per_class_ap": per_class_ap, "map": map_score}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute top-*k* classification accuracy.

    Args:
        logits: Raw model output tensor of shape (B, C).
        targets: Ground-truth class indices of shape (B,).
        k: Number of top predictions to consider.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if logits.shape[0] == 0:
        return 0.0
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices  # (B, k)
    correct = topk.eq(targets.unsqueeze(1).expand_as(topk)).any(dim=1)
    return float(correct.float().mean().item())
