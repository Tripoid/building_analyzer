"""Integration tests for the FastAPI backend.

Tests use lightweight stub inferencers injected via FastAPI's
``dependency_overrides`` so that no real ML models need to be loaded.
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.dependencies import (
    get_damage_inferencer,
    get_material_inferencer,
    get_pipeline,
    get_segmentation_inferencer,
)
from ml.common.base_model import (
    ClassificationResult,
    DetectionInstance,
    DetectionResult,
    SegmentationResult,
)
from ml.damage_detection.inference import DamageDetectionPrediction
from ml.facade_segmentation.inference import SegmentationPrediction
from ml.material_classification.inference import MaterialInferencer
from ml.pipeline.result import (
    BuildingAnalysisResult,
    DamageInstance,
    MaterialSummary,
    SegmentationSummary,
)

import torch


# ---------------------------------------------------------------------------
# Test fixtures — stub inferencers
# ---------------------------------------------------------------------------


def _make_jpeg_bytes(h: int = 64, w: int = 64) -> bytes:
    """Create a minimal valid JPEG image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_stub_seg_inferencer() -> MagicMock:
    inferencer = MagicMock()

    def _predict_from_array(image):
        h, w = image.shape[:2]
        mask = torch.zeros(32, 32, dtype=torch.long)
        probs = torch.ones(7, 32, 32) / 7.0
        pred = SegmentationPrediction(
            mask=mask,
            probabilities=probs,
            class_names=["background", "wall", "window", "door", "balcony", "cornice", "damaged"],
            original_size=(h, w),
        )
        return pred

    inferencer.predict_from_array.side_effect = _predict_from_array
    return inferencer


def _make_stub_det_inferencer() -> MagicMock:
    inferencer = MagicMock()
    inferencer.cfg = MagicMock()
    inferencer.cfg.score_threshold = 0.4
    inferencer.model = MagicMock()
    inferencer.model.class_names = ["crack", "spalling", "corrosion", "delamination", "efflorescence"]

    def _predict_from_array(image):
        result = DetectionResult(
            instances=[
                DetectionInstance(label=0, score=0.85, box=[10.0, 20.0, 50.0, 80.0])
            ]
        )
        return DamageDetectionPrediction(
            result=result,
            class_names=["crack", "spalling", "corrosion", "delamination", "efflorescence"],
            original_size=image.shape[:2],
            model_size=(64, 64),
        )

    inferencer.predict_from_array.side_effect = _predict_from_array
    return inferencer


def _make_stub_mat_inferencer() -> MagicMock:
    inferencer = MagicMock()

    def _predict_from_array(image):
        return ClassificationResult(
            label=0,
            label_name="concrete",
            score=0.92,
            scores={"concrete": 0.92, "brick": 0.05, "glass": 0.01,
                    "wood": 0.01, "metal": 0.01, "stone": 0.0},
        )

    def _predict_regions(image, boxes):
        return [
            {
                "box": box,
                "label": 0,
                "label_name": "concrete",
                "score": 0.88,
                "scores": {"concrete": 0.88},
            }
            for box in boxes
        ]

    inferencer.predict_from_array.side_effect = _predict_from_array
    inferencer.predict_regions.side_effect = _predict_regions
    return inferencer


def _make_stub_pipeline() -> MagicMock:
    pipeline = MagicMock()

    def _analyze(image):
        result = BuildingAnalysisResult()
        result.segmentation = SegmentationSummary(
            class_area_fractions={"background": 0.3, "wall": 0.6, "damaged": 0.1},
            dominant_class="wall",
            damaged_area_fraction=0.1,
        )
        result.damage_instances = [
            DamageInstance(
                label=0,
                label_name="crack",
                score=0.85,
                box=[10.0, 20.0, 50.0, 80.0],
                material_in_region="concrete",
                material_score=0.9,
            )
        ]
        result.materials = MaterialSummary(
            overall_dominant_material="concrete",
            intact_material="concrete",
            damaged_material="concrete",
        )
        result.metadata = {"elapsed_seconds": 0.1}
        return result

    pipeline.analyze.side_effect = _analyze
    return pipeline


@pytest.fixture()
def client() -> TestClient:
    """Return a test client with all ML dependencies stubbed out."""
    app.dependency_overrides[get_segmentation_inferencer] = lambda: _make_stub_seg_inferencer()
    app.dependency_overrides[get_damage_inferencer] = lambda: _make_stub_det_inferencer()
    app.dependency_overrides[get_material_inferencer] = lambda: _make_stub_mat_inferencer()
    app.dependency_overrides[get_pipeline] = lambda: _make_stub_pipeline()

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


# ---------------------------------------------------------------------------
# Segmentation endpoint tests
# ---------------------------------------------------------------------------


class TestSegmentationEndpoint:
    def test_predict_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/segmentation/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/segmentation/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "class_area_fractions" in body
        assert "dominant_class" in body
        assert "damaged_area_fraction" in body

    def test_predict_area_fractions_sum_to_one(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/segmentation/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        fractions = resp.json()["class_area_fractions"]
        total = sum(fractions.values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_with_mask(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/segmentation/predict?include_mask=true",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["colored_mask_b64"] is not None
        # Should be valid base64
        decoded = base64.b64decode(body["colored_mask_b64"])
        assert len(decoded) > 0

    def test_predict_invalid_file_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/segmentation/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Damage detection endpoint tests
# ---------------------------------------------------------------------------


class TestDamageDetectionEndpoint:
    def test_predict_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/damage/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/damage/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "num_detections" in body
        assert "detections" in body
        assert isinstance(body["detections"], list)

    def test_predict_detection_structure(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/damage/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert body["num_detections"] == len(body["detections"])
        for det in body["detections"]:
            assert "label" in det
            assert "label_name" in det
            assert "score" in det
            assert "box" in det
            box = det["box"]
            assert all(k in box for k in ("x1", "y1", "x2", "y2"))

    def test_score_threshold_query_param(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/damage/predict?score_threshold=0.95",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Material classification endpoint tests
# ---------------------------------------------------------------------------


class TestMaterialEndpoint:
    def test_predict_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/materials/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/materials/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "label" in body
        assert "label_name" in body
        assert "score" in body
        assert "scores" in body
        assert isinstance(body["scores"], dict)

    def test_predict_regions_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/materials/predict-regions?boxes=10,10,50,80&boxes=20,30,100,120",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert len(body["results"]) == 2

    def test_predict_regions_invalid_box(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/materials/predict-regions?boxes=bad",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pipeline endpoint tests
# ---------------------------------------------------------------------------


class TestPipelineEndpoint:
    def test_analyze_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pipeline/analyze",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_analyze_response_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pipeline/analyze",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "segmentation" in body
        assert "num_damage_instances" in body
        assert "damage_instances" in body
        assert "materials" in body
        assert "metadata" in body

    def test_analyze_segmentation_structure(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pipeline/analyze",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        seg = resp.json()["segmentation"]
        assert "class_area_fractions" in seg
        assert "dominant_class" in seg
        assert "damaged_area_fraction" in seg

    def test_analyze_damage_instances_structure(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pipeline/analyze",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert body["num_damage_instances"] == len(body["damage_instances"])
        for di in body["damage_instances"]:
            assert "label" in di
            assert "box" in di

    def test_analyze_materials_structure(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pipeline/analyze",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        mat = resp.json()["materials"]
        assert "overall_dominant_material" in mat
        assert "intact_material" in mat
        assert "damaged_material" in mat

    def test_analyze_invalid_file_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/pipeline/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422
