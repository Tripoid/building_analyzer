"""Pydantic response schemas for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class SegmentationResponse(BaseModel):
    """Response schema for ``POST /api/v1/segmentation/predict``."""

    class_area_fractions: dict[str, float] = Field(
        description="Fraction of pixels assigned to each class."
    )
    dominant_class: str = Field(description="Class with the highest pixel fraction.")
    damaged_area_fraction: float = Field(
        description="Fraction of pixels classified as 'damaged'."
    )
    colored_mask_b64: str | None = Field(
        default=None,
        description="Base-64 encoded PNG of the coloured segmentation mask.",
    )


# ---------------------------------------------------------------------------
# Damage detection
# ---------------------------------------------------------------------------


class DamageDetectionInstance(BaseModel):
    """A single detected damage region."""

    label: int
    label_name: str
    score: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class DamageDetectionResponse(BaseModel):
    """Response schema for ``POST /api/v1/damage/predict``."""

    num_detections: int
    detections: list[DamageDetectionInstance]


# ---------------------------------------------------------------------------
# Material classification
# ---------------------------------------------------------------------------


class MaterialClassificationResponse(BaseModel):
    """Response schema for ``POST /api/v1/materials/predict``."""

    label: int
    label_name: str
    score: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(
        description="Per-class probability scores."
    )


# ---------------------------------------------------------------------------
# Region-based material classification
# ---------------------------------------------------------------------------


class RegionMaterialResult(BaseModel):
    """Material classification result for one bounding-box region."""

    box: BoundingBox
    label: int
    label_name: str
    score: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float]


class RegionMaterialResponse(BaseModel):
    """Response schema for region-level material classification."""

    results: list[RegionMaterialResult]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class PipelineSegmentationSummary(BaseModel):
    class_area_fractions: dict[str, float]
    dominant_class: str
    damaged_area_fraction: float


class PipelineDamageInstance(BaseModel):
    label: int
    label_name: str
    score: float
    box: BoundingBox
    material_in_region: str | None = None
    material_score: float | None = None


class PipelineMaterialSummary(BaseModel):
    overall_dominant_material: str | None
    intact_material: str | None
    damaged_material: str | None


class PipelineAnalysisResponse(BaseModel):
    """Response schema for ``POST /api/v1/pipeline/analyze``."""

    image_path: str | None = None
    segmentation: PipelineSegmentationSummary
    num_damage_instances: int
    damage_instances: list[PipelineDamageInstance]
    materials: PipelineMaterialSummary
    metadata: dict


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
