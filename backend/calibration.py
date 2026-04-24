"""
Scale calibration: translate pixel measurements into real-world metres.

Flutter sends a calibrator reference on the photo — either two tapped endpoints
of a known dimension (preferred) or a bounding box. We compute:

    px_per_m_linear  — linear pixel density (px per 1 metre)
    m2_per_px        — area-per-pixel, used by the estimator

Priors guard against obvious input mistakes (e.g. a "door" reported as 5 m wide).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

ReferenceType = Literal["door", "window", "brick", "custom"]

_PRIORS: dict[ReferenceType, tuple[float, float]] = {
    "door": (0.7, 1.2),
    "window": (0.8, 2.0),
    "brick": (0.2, 0.3),
    "custom": (0.01, 100.0),
}


@dataclass
class ScaleCalibration:
    px_per_m_linear: float
    m2_per_px: float
    reference_type: ReferenceType
    reference_width_m: float
    reference_height_m: float | None
    warnings: list[str] = field(default_factory=list)

    def area_px_to_m2(self, area_px: int | float) -> float:
        return float(area_px) * self.m2_per_px

    def length_px_to_m(self, length_px: int | float) -> float:
        return float(length_px) / self.px_per_m_linear if self.px_per_m_linear else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _validate_prior(
    reference_type: ReferenceType, reference_width_m: float, warnings: list[str]
) -> None:
    lo, hi = _PRIORS.get(reference_type, _PRIORS["custom"])
    if not (lo <= reference_width_m <= hi):
        warnings.append(
            f"reference_width_m={reference_width_m}m outside prior "
            f"[{lo}, {hi}] for '{reference_type}' — калибровка может быть неточной"
        )


def calibrate_from_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
    reference_width_m: float,
    reference_type: ReferenceType = "door",
    reference_height_m: float | None = None,
) -> ScaleCalibration:
    """
    Two-tap calibration: user tapped both ends of a known dimension.

    The distance between p1 and p2 is the `reference_width_m` in pixels.
    If `reference_height_m` is also given, we assume the user approximately
    tapped the width, so area scale = width_m * height_m / (dist_px**2 * aspect),
    where `aspect = height/width`. In practice we compute a single linear scale
    and derive m2_per_px as 1/(px_per_m**2) * aspect_correction.
    """
    warnings: list[str] = []
    _validate_prior(reference_type, reference_width_m, warnings)

    dist_px = _distance(p1, p2)
    if dist_px < 5:
        raise ValueError("points are too close — tap two distinct ends of the reference")
    if reference_width_m <= 0:
        raise ValueError("reference_width_m must be > 0")

    px_per_m = dist_px / reference_width_m
    # Square pixels assumption → area scale is the inverse square.
    m2_per_px = 1.0 / (px_per_m * px_per_m)

    # If the user also provided a height, prefer the direct area-from-rect formula
    # (caller would have sent a bbox in that case; here we trust px_per_m).
    return ScaleCalibration(
        px_per_m_linear=px_per_m,
        m2_per_px=m2_per_px,
        reference_type=reference_type,
        reference_width_m=reference_width_m,
        reference_height_m=reference_height_m,
        warnings=warnings,
    )


def calibrate_from_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    reference_width_m: float,
    reference_type: ReferenceType = "door",
    reference_height_m: float | None = None,
) -> ScaleCalibration:
    """
    Rectangle-mode calibration: user drew a bbox around the reference object.

    - If both width_m and height_m are given → direct 2-D area scale.
    - Else → derive by the bbox side that represents the width; square assumption
      for the other axis.
    """
    warnings: list[str] = []
    _validate_prior(reference_type, reference_width_m, warnings)

    x1, y1, x2, y2 = bbox_xyxy
    width_px = max(1.0, abs(x2 - x1))
    height_px = max(1.0, abs(y2 - y1))

    if reference_width_m <= 0:
        raise ValueError("reference_width_m must be > 0")

    # Pixel-density from width axis
    px_per_m = width_px / reference_width_m

    if reference_height_m and reference_height_m > 0:
        # Genuine 2-D scale (handles non-square pixels / perspective)
        area_m2 = reference_width_m * reference_height_m
        area_px = width_px * height_px
        m2_per_px = area_m2 / area_px
    else:
        m2_per_px = 1.0 / (px_per_m * px_per_m)

    return ScaleCalibration(
        px_per_m_linear=px_per_m,
        m2_per_px=m2_per_px,
        reference_type=reference_type,
        reference_width_m=reference_width_m,
        reference_height_m=reference_height_m,
        warnings=warnings,
    )


def calibrate_from_mask(
    calibrator_mask: np.ndarray,
    reference_width_m: float,
    reference_height_m: float | None = None,
    reference_type: ReferenceType = "door",
) -> ScaleCalibration:
    """
    Mask-based calibration (advanced path): a binary mask of the reference
    object (e.g. a SAM segmentation of the door). We derive the bbox from
    the mask and forward to bbox calibration, but override m2_per_px with
    the true mask area when both dimensions are known.
    """
    warnings: list[str] = []
    _validate_prior(reference_type, reference_width_m, warnings)

    ys, xs = np.where(calibrator_mask > 0)
    if ys.size == 0:
        raise ValueError("empty calibrator mask")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = (x1, y1, x2, y2)

    result = calibrate_from_bbox(
        bbox, reference_width_m, reference_type, reference_height_m
    )

    # If both dims provided, prefer mask_area over bbox_area (real shape, not rect).
    if reference_height_m and reference_height_m > 0:
        mask_area_px = int(calibrator_mask.sum())
        if mask_area_px > 0:
            result.m2_per_px = (reference_width_m * reference_height_m) / mask_area_px

    result.warnings = warnings + result.warnings
    return result


def fallback_from_total_area(total_area_m2: float, total_px: int) -> ScaleCalibration:
    """
    Legacy fallback for when the client did not send a calibrator — matches the
    previous behaviour of the repair_calculator: evenly distribute total_area_m2
    over the silhouette pixel area. Marked with a warning.
    """
    if total_px <= 0 or total_area_m2 <= 0:
        raise ValueError("fallback requires total_area_m2 > 0 and total_px > 0")
    m2_per_px = total_area_m2 / total_px
    px_per_m = 1.0 / math.sqrt(m2_per_px) if m2_per_px > 0 else 0.0
    return ScaleCalibration(
        px_per_m_linear=px_per_m,
        m2_per_px=m2_per_px,
        reference_type="custom",
        reference_width_m=0.0,
        reference_height_m=None,
        warnings=[
            "No calibrator provided — scale estimated from total_area_m2. "
            "Предоставьте калибратор для точного масштаба."
        ],
    )
