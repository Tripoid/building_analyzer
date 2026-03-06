"""End-to-end building facade analysis pipeline.

The pipeline orchestrates three ML modules in sequence:

1. **Facade segmentation** — produces a per-pixel class map.
2. **Damage detection** — locates and scores damage instances.
3. **Material classification** — identifies materials in intact and damaged
   regions.

Each stage can be configured independently, and any stage can be disabled via
its ``enabled`` flag in :class:`PipelineConfig`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.damage_detection.inference import DamageDetectionInferencer, DamageDetectionInferencerConfig
from ml.facade_segmentation.inference import SegmentationInferencer, SegmentationInferencerConfig
from ml.material_classification.inference import MaterialInferencer, MaterialInferencerConfig
from ml.material_classification.utils import aggregate_region_materials
from ml.pipeline.result import (
    BuildingAnalysisResult,
    DamageInstance,
    MaterialSummary,
    SegmentationSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class SegmentationStageConfig:
    model_name: str = "unet"
    model_weights: str | None = None
    image_size: tuple[int, int] = (512, 512)
    tta: bool = False
    enabled: bool = True


@dataclass
class DamageStageConfig:
    model_name: str = "anchor_free"
    model_weights: str | None = None
    image_size: tuple[int, int] = (640, 640)
    score_threshold: float = 0.4
    nms_iou_threshold: float = 0.5
    max_detections: int = 100
    enabled: bool = True


@dataclass
class MaterialStageConfig:
    model_name: str = "cnn_classifier"
    model_weights: str | None = None
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 16
    classify_damage_regions: bool = True
    classify_intact_region: bool = True
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for the full analysis pipeline.

    Args:
        device: PyTorch device string (``"auto"``, ``"cpu"``, ``"cuda"``).
        segmentation: Configuration for the segmentation stage.
        damage: Configuration for the damage detection stage.
        materials: Configuration for the material classification stage.
    """

    device: str = "auto"
    segmentation: SegmentationStageConfig = field(default_factory=SegmentationStageConfig)
    damage: DamageStageConfig = field(default_factory=DamageStageConfig)
    materials: MaterialStageConfig = field(default_factory=MaterialStageConfig)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class BuildingAnalysisPipeline:
    """Orchestrates all ML stages for building facade analysis.

    Stages are instantiated lazily (on first call to :meth:`analyze`) so that
    the pipeline object can be constructed and configured without loading all
    model weights into memory immediately.

    Example::

        config = PipelineConfig(device="cpu")
        pipeline = BuildingAnalysisPipeline(config)
        result = pipeline.analyze_from_path("facade.jpg")
        print(result.to_dict())

    Args:
        config: Full pipeline configuration.
        seg_model: Optional pre-built segmentation inferencer (bypasses
            ``config.segmentation`` if provided).
        det_model: Optional pre-built damage detection inferencer.
        mat_model: Optional pre-built material classification inferencer.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        seg_model: SegmentationInferencer | None = None,
        det_model: DamageDetectionInferencer | None = None,
        mat_model: MaterialInferencer | None = None,
    ) -> None:
        self.cfg = config or PipelineConfig()
        self._seg: SegmentationInferencer | None = seg_model
        self._det: DamageDetectionInferencer | None = det_model
        self._mat: MaterialInferencer | None = mat_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_from_path(self, path: str | Path) -> BuildingAnalysisResult:
        """Load an image from *path* and run the full pipeline.

        Args:
            path: Path to an image file.

        Returns:
            A :class:`~ml.pipeline.result.BuildingAnalysisResult`.
        """
        from PIL import Image as _Image
        image = np.array(_Image.open(path).convert("RGB"), dtype=np.uint8)
        result = self.analyze(image)
        result.image_path = str(path)
        return result

    def analyze(self, image: np.ndarray) -> BuildingAnalysisResult:
        """Run all enabled pipeline stages on a single HWC uint8 RGB image.

        Args:
            image: RGB uint8 array of shape (H, W, 3).

        Returns:
            A :class:`~ml.pipeline.result.BuildingAnalysisResult`.
        """
        t0 = time.time()
        result = BuildingAnalysisResult()
        result.metadata["image_shape"] = list(image.shape)

        # ----------------------------------------------------------------
        # Stage 1 — Facade segmentation
        # ----------------------------------------------------------------
        seg_mask = None
        if self.cfg.segmentation.enabled:
            seg_pred = self._get_seg().predict_from_array(image)
            seg_mask = seg_pred.mask.numpy()  # (H_model, W_model)

            fractions = seg_pred.class_area_fractions()
            dominant = max(fractions, key=fractions.get)
            damaged_frac = fractions.get("damaged", 0.0)

            result.segmentation = SegmentationSummary(
                class_area_fractions=fractions,
                dominant_class=dominant,
                damaged_area_fraction=damaged_frac,
            )
            logger.info(
                "Segmentation: dominant=%s  damaged_frac=%.3f", dominant, damaged_frac
            )

        # ----------------------------------------------------------------
        # Stage 2 — Damage detection
        # ----------------------------------------------------------------
        damage_boxes: list[list[float]] = []
        if self.cfg.damage.enabled:
            det_pred = self._get_det().predict_from_array(image)
            for inst in det_pred.result.instances:
                result.damage_instances.append(
                    DamageInstance(
                        label=inst.label,
                        label_name=(
                            self.cfg.damage.model_name  # placeholder; real name in inferencer
                        ),
                        score=inst.score,
                        box=inst.box,
                    )
                )
                damage_boxes.append(inst.box)

            # Fix label names using the damage inferencer's class list
            det_inferencer = self._get_det()
            for i, inst in enumerate(result.damage_instances):
                if inst.label < len(det_inferencer.model.class_names):
                    result.damage_instances[i].label_name = det_inferencer.model.class_names[
                        inst.label
                    ]
            logger.info("Damage detection: %d instances", len(result.damage_instances))

        # ----------------------------------------------------------------
        # Stage 3 — Material classification
        # ----------------------------------------------------------------
        if self.cfg.materials.enabled:
            mat_inferencer = self._get_mat()

            # 3a. Classify damaged regions
            if self.cfg.materials.classify_damage_regions and damage_boxes:
                region_results = mat_inferencer.predict_regions(image, damage_boxes)
                for i, (di, reg_r) in enumerate(
                    zip(result.damage_instances, region_results)
                ):
                    result.damage_instances[i].material_in_region = reg_r["label_name"]
                    result.damage_instances[i].material_score = reg_r["score"]

            # 3b. Classify intact region (whole image as proxy)
            intact_result = None
            if self.cfg.materials.classify_intact_region:
                intact_result = mat_inferencer.predict_from_array(image)

            # Aggregate material summary
            region_dicts = []
            for di in result.damage_instances:
                if di.material_in_region:
                    region_dicts.append(
                        {
                            "label_name": di.material_in_region,
                            "score": di.material_score or 0.0,
                            "scores": {},
                        }
                    )

            summary = aggregate_region_materials(region_dicts)
            result.materials = MaterialSummary(
                overall_dominant_material=summary.get("dominant_material"),
                intact_material=(intact_result.label_name if intact_result else None),
                damaged_material=summary.get("dominant_material"),
                region_results=region_dicts,
            )
            logger.info(
                "Materials: intact=%s  damaged=%s",
                result.materials.intact_material,
                result.materials.damaged_material,
            )

        result.metadata["elapsed_seconds"] = round(time.time() - t0, 3)
        return result

    def analyze_batch(self, images: list[np.ndarray]) -> list[BuildingAnalysisResult]:
        """Run the pipeline on a list of images.

        Args:
            images: List of RGB uint8 arrays.

        Returns:
            List of :class:`~ml.pipeline.result.BuildingAnalysisResult`.
        """
        return [self.analyze(img) for img in images]

    # ------------------------------------------------------------------
    # Lazy inferencer accessors
    # ------------------------------------------------------------------

    def _get_seg(self) -> SegmentationInferencer:
        if self._seg is None:
            cfg = self.cfg.segmentation
            self._seg = SegmentationInferencer.from_config(
                SegmentationInferencerConfig(
                    model_name=cfg.model_name,
                    model_weights=cfg.model_weights,
                    device=self.cfg.device,
                    image_size=cfg.image_size,
                    tta=cfg.tta,
                )
            )
        return self._seg

    def _get_det(self) -> DamageDetectionInferencer:
        if self._det is None:
            cfg = self.cfg.damage
            self._det = DamageDetectionInferencer.from_config(
                DamageDetectionInferencerConfig(
                    model_name=cfg.model_name,
                    model_weights=cfg.model_weights,
                    device=self.cfg.device,
                    image_size=cfg.image_size,
                    score_threshold=cfg.score_threshold,
                    nms_iou_threshold=cfg.nms_iou_threshold,
                    max_detections=cfg.max_detections,
                )
            )
        return self._det

    def _get_mat(self) -> MaterialInferencer:
        if self._mat is None:
            cfg = self.cfg.materials
            self._mat = MaterialInferencer.from_config(
                MaterialInferencerConfig(
                    model_name=cfg.model_name,
                    model_weights=cfg.model_weights,
                    device=self.cfg.device,
                    image_size=cfg.image_size,
                    batch_size=cfg.batch_size,
                )
            )
        return self._mat
