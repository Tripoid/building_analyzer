"""FastAPI dependency injection for ML model inferencers.

Each inferencer is constructed lazily on first use and cached as a
module-level singleton.  This avoids the overhead of rebuilding models on
every request while remaining testable (dependencies can be overridden via
``app.dependency_overrides``).
"""

from __future__ import annotations

from functools import lru_cache

from backend.config import get_settings
from ml.damage_detection.inference import DamageDetectionInferencer, DamageDetectionInferencerConfig
from ml.facade_segmentation.inference import SegmentationInferencer, SegmentationInferencerConfig
from ml.material_classification.inference import MaterialInferencer, MaterialInferencerConfig
from ml.pipeline.pipeline import (
    BuildingAnalysisPipeline,
    DamageStageConfig,
    MaterialStageConfig,
    PipelineConfig,
    SegmentationStageConfig,
)


@lru_cache(maxsize=1)
def get_segmentation_inferencer() -> SegmentationInferencer:
    """Return (and cache) the facade segmentation inferencer."""
    settings = get_settings()
    return SegmentationInferencer.from_config(
        SegmentationInferencerConfig(
            model_name=settings.seg_model_name,
            model_weights=settings.seg_model_weights,
            device=settings.device,
            image_size=settings.seg_image_size,
            tta=settings.seg_tta,
        )
    )


@lru_cache(maxsize=1)
def get_damage_inferencer() -> DamageDetectionInferencer:
    """Return (and cache) the damage detection inferencer."""
    settings = get_settings()
    return DamageDetectionInferencer.from_config(
        DamageDetectionInferencerConfig(
            model_name=settings.det_model_name,
            model_weights=settings.det_model_weights,
            device=settings.device,
            image_size=settings.det_image_size,
            score_threshold=settings.det_score_threshold,
            nms_iou_threshold=settings.det_nms_iou_threshold,
            max_detections=settings.det_max_detections,
        )
    )


@lru_cache(maxsize=1)
def get_material_inferencer() -> MaterialInferencer:
    """Return (and cache) the material classification inferencer."""
    settings = get_settings()
    return MaterialInferencer.from_config(
        MaterialInferencerConfig(
            model_name=settings.mat_model_name,
            model_weights=settings.mat_model_weights,
            device=settings.device,
            image_size=settings.mat_image_size,
            batch_size=settings.mat_batch_size,
        )
    )


@lru_cache(maxsize=1)
def get_pipeline() -> BuildingAnalysisPipeline:
    """Return (and cache) the full analysis pipeline."""
    settings = get_settings()
    cfg = PipelineConfig(
        device=settings.device,
        segmentation=SegmentationStageConfig(
            model_name=settings.seg_model_name,
            model_weights=settings.seg_model_weights,
            image_size=settings.seg_image_size,
            tta=settings.seg_tta,
        ),
        damage=DamageStageConfig(
            model_name=settings.det_model_name,
            model_weights=settings.det_model_weights,
            image_size=settings.det_image_size,
            score_threshold=settings.det_score_threshold,
            nms_iou_threshold=settings.det_nms_iou_threshold,
            max_detections=settings.det_max_detections,
        ),
        materials=MaterialStageConfig(
            model_name=settings.mat_model_name,
            model_weights=settings.mat_model_weights,
            image_size=settings.mat_image_size,
            batch_size=settings.mat_batch_size,
        ),
    )
    # Share the already-cached inferencers with the pipeline
    return BuildingAnalysisPipeline(
        config=cfg,
        seg_model=get_segmentation_inferencer(),
        det_model=get_damage_inferencer(),
        mat_model=get_material_inferencer(),
    )
