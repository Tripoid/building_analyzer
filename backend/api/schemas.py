"""
Pydantic v2 request/response models for the public API.

Schemas are kept transport-only — business types live in calibration.py,
ml_pipeline.py, and estimator/.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─────────────── Health ───────────────


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: bool
    device: str
    version: str
    inpaint_provider: str
    scraper_enabled: bool


# ─────────────── Calibration ───────────────


Point = tuple[float, float]
Bbox = tuple[float, float, float, float]


class CalibrationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_width_px: int = Field(..., gt=0)
    image_height_px: int = Field(..., gt=0)
    reference_type: Literal["door", "window", "brick", "custom"] = "door"
    reference_width_m: float = Field(..., gt=0, le=1000)
    reference_height_m: Optional[float] = Field(default=None, gt=0, le=1000)

    # Two-tap (preferred) OR bbox — exactly one must be provided.
    p1: Optional[Point] = None
    p2: Optional[Point] = None
    bbox: Optional[Bbox] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "CalibrationRequest":
        has_points = self.p1 is not None and self.p2 is not None
        has_bbox = self.bbox is not None
        if has_points == has_bbox:
            raise ValueError("provide EITHER p1+p2 OR bbox")
        return self


class CalibrationResponse(BaseModel):
    calibration_id: str
    px_per_m: float
    m2_per_px: float
    reference_type: str
    reference_width_m: float
    reference_height_m: Optional[float] = None
    warnings: list[str] = []


# ─────────────── Analysis ───────────────


class DamageItem(BaseModel):
    type: str
    type_display: str
    percentage: float
    area_px: int
    area_m2: float = 0.0
    severity: Literal["low", "medium", "high"]
    severity_display: str
    affected_layers: list[str] = []
    crack_depth: Optional[str] = None


class MaterialItem(BaseModel):
    name: str
    name_display: str
    percentage: float
    area_px: int
    area_m2: float = 0.0


class LayerAnalysisItem(BaseModel):
    area_px: int
    area_m2: float = 0.0
    affected_layers: list[str]
    crack_depth: Optional[str] = None


class MaskUrls(BaseModel):
    """URLs of layered PNG masks the Flutter viewer stacks on top of the photo."""

    base_image: str
    geometry: dict[str, str] = {}       # window, door, balcony, roof, ...
    materials: dict[str, str] = {}      # brick, wood, metal, glass, ...
    defects: dict[str, str] = {}        # crack, peeling, mold, ...
    visualizations: dict[str, str] = {}  # heatmap, segments, overlay, defects


class AnalysisResponse(BaseModel):
    id: str
    overall_score: float
    overall_condition: str
    total_area_px: int
    total_area_m2: float
    damaged_area_px: int
    damaged_area_m2: float
    calibration_id: Optional[str] = None
    calibration_warnings: list[str] = []
    damages: list[DamageItem]
    materials: list[MaterialItem]
    layer_analysis: dict[str, LayerAnalysisItem] = {}
    processed_images: list[str] = []
    masks: MaskUrls
    price_snapshot_date: Optional[str] = None
    price_source: Optional[str] = None
    repair_estimate: dict[str, Any] = {}


# ─────────────── Estimate ───────────────


class EstimateRequest(BaseModel):
    analysis_id: str
    waste_factor: float = Field(1.10, ge=1.0, le=2.0)
    vat_rate: float = Field(0.20, ge=0.0, le=1.0)


class EstimateResponse(BaseModel):
    analysis_id: str
    repair_estimate: dict[str, Any]
    price_snapshot_date: Optional[str] = None
    price_source: str  # "live" | "yaml_fallback"
    stale: bool = False


# ─────────────── Restoration ───────────────


class RestoreRequest(BaseModel):
    quality: Literal["fast", "high"] = "fast"
    prompt: Optional[str] = None


class RestoreResponse(BaseModel):
    analysis_id: str
    restored_url: str
    provider: str
    duration_ms: int
