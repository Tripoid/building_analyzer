"""Integration tests for the end-to-end building analysis pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ml.damage_detection.inference import DamageDetectionInferencer, DamageDetectionInferencerConfig
from ml.damage_detection.model import AnchorFreeDamageDetector
from ml.facade_segmentation.inference import SegmentationInferencer, SegmentationInferencerConfig
from ml.facade_segmentation.model import UNetSegmentation
from ml.material_classification.inference import MaterialInferencer, MaterialInferencerConfig
from ml.material_classification.model import CNNMaterialClassifier
from ml.pipeline.pipeline import (
    BuildingAnalysisPipeline,
    DamageStageConfig,
    MaterialStageConfig,
    PipelineConfig,
    SegmentationStageConfig,
)
from ml.pipeline.result import BuildingAnalysisResult, DamageInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_SEG_CLASSES = 7  # facade segmentation classes
NUM_DET_CLASSES = 5  # damage categories
NUM_MAT_CLASSES = 6  # material classes

SEG_CLASS_NAMES = ["background", "wall", "window", "door", "balcony", "cornice", "damaged"]
DET_CLASS_NAMES = ["crack", "spalling", "corrosion", "delamination", "efflorescence"]
MAT_CLASS_NAMES = ["concrete", "brick", "glass", "wood", "metal", "stone"]


def _make_pipeline(
    seg_enabled: bool = True,
    det_enabled: bool = True,
    mat_enabled: bool = True,
) -> BuildingAnalysisPipeline:
    """Build a lightweight pipeline with tiny models for testing."""
    seg_model = UNetSegmentation(
        num_classes=NUM_SEG_CLASSES, class_names=SEG_CLASS_NAMES, base_channels=8
    )
    seg_inferencer = SegmentationInferencer(
        model=seg_model,
        config=SegmentationInferencerConfig(device="cpu", image_size=(64, 64)),
    )

    det_model = AnchorFreeDamageDetector(
        num_classes=NUM_DET_CLASSES, class_names=DET_CLASS_NAMES, fpn_channels=32
    )
    det_inferencer = DamageDetectionInferencer(
        model=det_model,
        config=DamageDetectionInferencerConfig(
            device="cpu", image_size=(64, 64), score_threshold=0.0
        ),
    )

    mat_model = CNNMaterialClassifier(
        num_classes=NUM_MAT_CLASSES, class_names=MAT_CLASS_NAMES, base_channels=8
    )
    mat_inferencer = MaterialInferencer(
        model=mat_model,
        config=MaterialInferencerConfig(device="cpu", image_size=(32, 32)),
    )

    cfg = PipelineConfig(
        device="cpu",
        segmentation=SegmentationStageConfig(enabled=seg_enabled),
        damage=DamageStageConfig(enabled=det_enabled, score_threshold=0.0),
        materials=MaterialStageConfig(enabled=mat_enabled),
    )

    return BuildingAnalysisPipeline(
        config=cfg,
        seg_model=seg_inferencer,
        det_model=det_inferencer,
        mat_model=mat_inferencer,
    )


def _random_image(h: int = 128, w: int = 128) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Basic pipeline tests
# ---------------------------------------------------------------------------


class TestBuildingAnalysisPipeline:
    def test_analyze_returns_result_object(self) -> None:
        pipeline = _make_pipeline()
        image = _random_image()
        result = pipeline.analyze(image)
        assert isinstance(result, BuildingAnalysisResult)

    def test_result_has_segmentation(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.analyze(_random_image())
        assert result.segmentation is not None
        assert isinstance(result.segmentation.class_area_fractions, dict)
        total = sum(result.segmentation.class_area_fractions.values())
        assert abs(total - 1.0) < 1e-5, f"Area fractions should sum to 1, got {total}"

    def test_result_has_damage_instances(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.analyze(_random_image())
        assert isinstance(result.damage_instances, list)
        for inst in result.damage_instances:
            assert isinstance(inst, DamageInstance)
            assert isinstance(inst.box, list)
            assert len(inst.box) == 4

    def test_result_has_materials(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.analyze(_random_image())
        assert result.materials is not None
        # intact_material should be set
        assert result.materials.intact_material in MAT_CLASS_NAMES

    def test_to_dict_is_json_serialisable(self) -> None:
        import json
        pipeline = _make_pipeline()
        result = pipeline.analyze(_random_image())
        d = result.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert len(serialised) > 0

    def test_metadata_contains_elapsed(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.analyze(_random_image())
        assert "elapsed_seconds" in result.metadata
        assert result.metadata["elapsed_seconds"] >= 0

    def test_analyze_from_path(self, tmp_path) -> None:
        from PIL import Image as _Image
        img = _Image.fromarray(_random_image())
        path = tmp_path / "facade.jpg"
        img.save(path)

        pipeline = _make_pipeline()
        result = pipeline.analyze_from_path(path)
        assert result.image_path == str(path)

    def test_analyze_batch(self) -> None:
        pipeline = _make_pipeline()
        images = [_random_image() for _ in range(3)]
        results = pipeline.analyze_batch(images)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, BuildingAnalysisResult)


# ---------------------------------------------------------------------------
# Disabled-stage tests
# ---------------------------------------------------------------------------


class TestPipelineDisabledStages:
    def test_disable_segmentation(self) -> None:
        pipeline = _make_pipeline(seg_enabled=False)
        result = pipeline.analyze(_random_image())
        # Segmentation summary should be default (empty fractions)
        assert result.segmentation.class_area_fractions == {}

    def test_disable_damage_detection(self) -> None:
        pipeline = _make_pipeline(det_enabled=False)
        result = pipeline.analyze(_random_image())
        assert result.damage_instances == []

    def test_disable_materials(self) -> None:
        pipeline = _make_pipeline(mat_enabled=False)
        result = pipeline.analyze(_random_image())
        assert result.materials.intact_material is None
        assert result.materials.damaged_material is None

    def test_disable_all_stages(self) -> None:
        pipeline = _make_pipeline(seg_enabled=False, det_enabled=False, mat_enabled=False)
        result = pipeline.analyze(_random_image())
        assert isinstance(result, BuildingAnalysisResult)


# ---------------------------------------------------------------------------
# Result dataclass tests
# ---------------------------------------------------------------------------


class TestBuildingAnalysisResult:
    def test_num_damage_instances(self) -> None:
        result = BuildingAnalysisResult()
        assert result.num_damage_instances == 0
        result.damage_instances.append(
            DamageInstance(label=0, label_name="crack", score=0.9, box=[0, 0, 10, 10])
        )
        assert result.num_damage_instances == 1

    def test_to_dict_structure(self) -> None:
        result = BuildingAnalysisResult(image_path="test.jpg")
        d = result.to_dict()
        assert d["image_path"] == "test.jpg"
        assert "segmentation" in d
        assert "damage_instances" in d
        assert "materials" in d
        assert "metadata" in d

    def test_to_dict_includes_damage_instance_fields(self) -> None:
        result = BuildingAnalysisResult()
        result.damage_instances.append(
            DamageInstance(
                label=0,
                label_name="crack",
                score=0.95,
                box=[10, 20, 50, 80],
                material_in_region="concrete",
                material_score=0.8,
            )
        )
        d = result.to_dict()
        inst = d["damage_instances"][0]
        assert inst["label"] == 0
        assert inst["label_name"] == "crack"
        assert inst["score"] == pytest.approx(0.95)
        assert inst["box"] == [10, 20, 50, 80]
        assert inst["material_in_region"] == "concrete"
