"""Damage detection model implementations.

Two concrete detectors are provided:

* :class:`AnchorFreeDamageDetector` — a FCOS-inspired anchor-free detector
  that predicts damage instances from a shared feature pyramid.
* :class:`TwoStageDetector` — a Region Proposal Network + ROI head (Faster
  RCNN-inspired) detector.

Both are registered with :class:`~ml.common.registry.ModelRegistry` and
share the :class:`~ml.common.base_model.BaseDetectionModel` interface.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.common.base_model import BaseDetectionModel
from ml.common.registry import ModelRegistry
from ml.damage_detection.dataset import DAMAGE_CLASS_NAMES


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _FPN(nn.Module):
    """Minimal Feature Pyramid Network operating on a simple backbone."""

    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.outputs = nn.ModuleList(
            [_ConvBnRelu(out_channels, out_channels, 3) for _ in in_channels]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Top-down feature fusion."""
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        # Top-down path
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode="nearest"
            )
        return [out(lat) for out, lat in zip(self.outputs, laterals)]


class _SimpleBackbone(nn.Module):
    """Lightweight multi-scale backbone for damage detection."""

    OUT_CHANNELS = [64, 128, 256]

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(_ConvBnRelu(3, 32), _ConvBnRelu(32, 64))
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2), _ConvBnRelu(64, 64), _ConvBnRelu(64, 128)
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2, 2), _ConvBnRelu(128, 128), _ConvBnRelu(128, 256)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


# ---------------------------------------------------------------------------
# Anchor-free detector (FCOS-style)
# ---------------------------------------------------------------------------


@ModelRegistry.register("anchor_free", namespace="detection")
class AnchorFreeDamageDetector(BaseDetectionModel):
    """FCOS-inspired anchor-free damage detector.

    Predicts per-location class scores, centerness, and LTRB box offsets on
    each feature pyramid level.  Post-processing (score thresholding + NMS)
    is handled by :meth:`predict`.

    Args:
        num_classes: Number of damage categories (excluding background).
        class_names: Ordered list of class names.
        fpn_channels: Number of channels in the FPN output.
    """

    def __init__(
        self,
        num_classes: int = len(DAMAGE_CLASS_NAMES),
        class_names: list[str] = DAMAGE_CLASS_NAMES,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names)

        self.backbone = _SimpleBackbone()
        self.fpn = _FPN(in_channels=_SimpleBackbone.OUT_CHANNELS, out_channels=fpn_channels)

        # Shared head towers
        self.cls_tower = nn.Sequential(*[_ConvBnRelu(fpn_channels, fpn_channels) for _ in range(4)])
        self.reg_tower = nn.Sequential(*[_ConvBnRelu(fpn_channels, fpn_channels) for _ in range(4)])

        # Output heads
        self.cls_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.reg_head = nn.Conv2d(fpn_channels, 4, 1)  # LTRB
        self.ctr_head = nn.Conv2d(fpn_channels, 1, 1)  # centerness

        # Strides per FPN level — must match backbone downsampling
        self.strides = [1, 2, 4]

    def forward(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Run the detector and return raw per-image prediction dicts.

        Returned dict keys:
        - ``"boxes"`` — float32 (N, 4) in pixel coords ``[x1, y1, x2, y2]``.
        - ``"labels"`` — int64 (N,) 0-based class indices.
        - ``"scores"`` — float32 (N,) combined cls × centerness scores.
        """
        b, _, h, w = images.shape
        features = self.backbone(images)
        fpn_feats = self.fpn(features)

        all_boxes: list[list[torch.Tensor]] = [[] for _ in range(b)]
        all_labels: list[list[torch.Tensor]] = [[] for _ in range(b)]
        all_scores: list[list[torch.Tensor]] = [[] for _ in range(b)]

        for feat, stride in zip(fpn_feats, self.strides):
            cls_out = torch.sigmoid(self.cls_head(self.cls_tower(feat)))  # (B, C, H', W')
            reg_out = F.relu(self.reg_head(self.reg_tower(feat)))  # (B, 4, H', W') — LTRB
            ctr_out = torch.sigmoid(self.ctr_head(self.reg_tower(feat)))  # (B, 1, H', W')

            fh, fw = feat.shape[2:]
            # Build grid of centre coordinates
            ys = (torch.arange(fh, device=feat.device).float() + 0.5) * stride
            xs = (torch.arange(fw, device=feat.device).float() + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H', W')

            for img_i in range(b):
                scores_map = cls_out[img_i] * ctr_out[img_i]  # (C, H', W')
                max_scores, pred_labels = scores_map.max(dim=0)  # (H', W')

                # Decode LTRB to [x1, y1, x2, y2]
                ltrb = reg_out[img_i]  # (4, H', W')
                x1 = (grid_x - ltrb[0]).clamp(min=0, max=w)
                y1 = (grid_y - ltrb[1]).clamp(min=0, max=h)
                x2 = (grid_x + ltrb[2]).clamp(min=0, max=w)
                y2 = (grid_y + ltrb[3]).clamp(min=0, max=h)

                boxes_lvl = torch.stack(
                    [x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten()], dim=-1
                )  # (H'*W', 4)
                all_boxes[img_i].append(boxes_lvl)
                all_labels[img_i].append(pred_labels.flatten())
                all_scores[img_i].append(max_scores.flatten())

        results: list[dict[str, torch.Tensor]] = []
        for img_i in range(b):
            if all_boxes[img_i]:
                results.append(
                    {
                        "boxes": torch.cat(all_boxes[img_i], dim=0),
                        "labels": torch.cat(all_labels[img_i], dim=0),
                        "scores": torch.cat(all_scores[img_i], dim=0),
                    }
                )
            else:
                results.append(
                    {
                        "boxes": torch.zeros(0, 4),
                        "labels": torch.zeros(0, dtype=torch.long),
                        "scores": torch.zeros(0),
                    }
                )
        return results


@ModelRegistry.register("two_stage", namespace="detection")
class TwoStageDetector(BaseDetectionModel):
    """Simplified two-stage (RPN + ROI head) damage detector.

    Implements a minimal Region Proposal Network that generates candidate
    anchors, followed by ROI pooling and a classification head.  Production
    use-cases should replace this with a torchvision ``FasterRCNN`` model.

    Args:
        num_classes: Number of damage categories (excluding background).
        class_names: Ordered list of class names.
        anchor_sizes: List of anchor sizes (pixels) to use per location.
    """

    def __init__(
        self,
        num_classes: int = len(DAMAGE_CLASS_NAMES),
        class_names: list[str] = DAMAGE_CLASS_NAMES,
        anchor_sizes: list[int] | None = None,
    ) -> None:
        super().__init__(num_classes=num_classes, class_names=class_names)
        self.anchor_sizes = anchor_sizes or [32, 64, 128]

        self.backbone = _SimpleBackbone()
        ch = _SimpleBackbone.OUT_CHANNELS[-1]

        # RPN
        self.rpn_conv = _ConvBnRelu(ch, ch)
        num_anchors = len(self.anchor_sizes)
        self.rpn_cls = nn.Conv2d(ch, num_anchors * 2, 1)
        self.rpn_reg = nn.Conv2d(ch, num_anchors * 4, 1)

        # ROI head (operates on fixed 7×7 pooled regions)
        self.roi_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.cls_out = nn.Linear(1024, num_classes + 1)  # +1 background
        self.reg_out = nn.Linear(1024, 4)

    def forward(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return raw per-image prediction dicts (same interface as anchor-free)."""
        b = images.shape[0]
        features = self.backbone(images)
        feat = features[-1]  # use deepest level

        # RPN forward (simplified: no anchor matching, just produces grid proposals)
        rpn_feat = self.rpn_conv(feat)
        rpn_cls_out = self.rpn_cls(rpn_feat)  # (B, 2*A, H', W')
        rpn_reg_out = self.rpn_reg(rpn_feat)  # (B, 4*A, H', W')

        # For inference, generate dummy proposals from the RPN outputs
        results: list[dict[str, torch.Tensor]] = []
        for img_i in range(b):
            # Flatten RPN outputs to get proposals
            cls_flat = rpn_cls_out[img_i].reshape(-1, 2)
            reg_flat = rpn_reg_out[img_i].reshape(-1, 4)
            fg_scores = torch.softmax(cls_flat, dim=1)[:, 1]

            # Generate anchor grid
            fh, fw = feat.shape[2:]
            stride = images.shape[2] // fh
            anchors = self._generate_anchors(fh, fw, stride, feat.device)

            # Apply regression deltas
            boxes = self._apply_deltas(anchors, reg_flat.detach())
            boxes = boxes.clamp(min=0)

            results.append(
                {
                    "boxes": boxes,
                    "labels": torch.zeros(boxes.shape[0], dtype=torch.long),
                    "scores": fg_scores.detach(),
                }
            )
        return results

    def _generate_anchors(
        self, fh: int, fw: int, stride: int, device: torch.device
    ) -> torch.Tensor:
        """Generate a flat list of anchor boxes for one feature level."""
        cy = (torch.arange(fh, device=device).float() + 0.5) * stride
        cx = (torch.arange(fw, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
        centres = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (N, 2)

        all_anchors = []
        for size in self.anchor_sizes:
            half = size / 2.0
            boxes = torch.stack(
                [
                    centres[:, 0] - half,
                    centres[:, 1] - half,
                    centres[:, 0] + half,
                    centres[:, 1] + half,
                ],
                dim=-1,
            )
            all_anchors.append(boxes)
        return torch.cat(all_anchors, dim=0)  # (N*A, 4)

    @staticmethod
    def _apply_deltas(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Decode predicted regression deltas into absolute boxes."""
        # Ensure matching size (deltas may be shorter due to RPN head shape)
        n = min(anchors.shape[0], deltas.shape[0])
        anchors = anchors[:n]
        deltas = deltas[:n]

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        cx = anchors[:, 0] + 0.5 * widths
        cy = anchors[:, 1] + 0.5 * heights

        pred_cx = deltas[:, 0] * widths + cx
        pred_cy = deltas[:, 1] * heights + cy
        pred_w = widths * torch.exp(deltas[:, 2].clamp(max=4))
        pred_h = heights * torch.exp(deltas[:, 3].clamp(max=4))

        return torch.stack(
            [
                pred_cx - 0.5 * pred_w,
                pred_cy - 0.5 * pred_h,
                pred_cx + 0.5 * pred_w,
                pred_cy + 0.5 * pred_h,
            ],
            dim=-1,
        )


# Default alias
DamageDetectionModel = AnchorFreeDamageDetector
