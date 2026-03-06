"""Unit and integration tests for the damage detection module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.common.base_model import DetectionInstance, DetectionResult
from ml.common.metrics import compute_map
from ml.common.registry import ModelRegistry
from ml.damage_detection.dataset import DAMAGE_CLASS_NAMES
from ml.damage_detection.inference import DamageDetectionInferencer, DamageDetectionInferencerConfig
from ml.damage_detection.model import AnchorFreeDamageDetector, TwoStageDetector
from ml.damage_detection.utils import apply_nms, compute_box_iou_matrix, draw_detections, scale_boxes_to_original

NUM_CLASSES = len(DAMAGE_CLASS_NAMES)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestAnchorFreeDamageDetector:
    def _make_model(self) -> AnchorFreeDamageDetector:
        return AnchorFreeDamageDetector(
            num_classes=NUM_CLASSES, class_names=DAMAGE_CLASS_NAMES, fpn_channels=64
        )

    def test_forward_output_structure(self) -> None:
        model = self._make_model()
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == 2
        for out in outputs:
            assert "boxes" in out
            assert "labels" in out
            assert "scores" in out

    def test_predict_returns_detection_results(self) -> None:
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64)
        results = model.predict(x, score_threshold=0.0)
        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)

    def test_predict_with_high_threshold_reduces_instances(self) -> None:
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64)
        results_low = model.predict(x, score_threshold=0.0)
        results_high = model.predict(x, score_threshold=0.99)
        assert len(results_high[0].instances) <= len(results_low[0].instances)

    def test_save_load(self, tmp_path) -> None:
        model = self._make_model()
        path = tmp_path / "detector.pt"
        model.save(path)
        model2 = self._make_model()
        model2.load(path)
        x = torch.randn(1, 3, 64, 64)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)[0]
            out2 = model2(x)[0]
        assert torch.allclose(out1["scores"], out2["scores"])


class TestTwoStageDetector:
    def _make_model(self) -> TwoStageDetector:
        return TwoStageDetector(
            num_classes=NUM_CLASSES, class_names=DAMAGE_CLASS_NAMES, anchor_sizes=[16, 32]
        )

    def test_forward_output_structure(self) -> None:
        model = self._make_model()
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == 2
        for out in outputs:
            assert "boxes" in out

    def test_predict_returns_detection_results(self) -> None:
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64)
        results = model.predict(x, score_threshold=0.0)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestDetectionRegistry:
    def test_anchor_free_registered(self) -> None:
        assert "anchor_free" in ModelRegistry.list_models(namespace="detection")

    def test_two_stage_registered(self) -> None:
        assert "two_stage" in ModelRegistry.list_models(namespace="detection")

    def test_build_anchor_free(self) -> None:
        model = ModelRegistry.build(
            "anchor_free",
            namespace="detection",
            num_classes=NUM_CLASSES,
            class_names=DAMAGE_CLASS_NAMES,
        )
        assert isinstance(model, AnchorFreeDamageDetector)


# ---------------------------------------------------------------------------
# NMS tests
# ---------------------------------------------------------------------------


class TestNMS:
    def test_apply_nms_filters_low_scores(self) -> None:
        boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.1])
        kb, kl, ks = apply_nms(boxes, labels, scores, score_threshold=0.5)
        assert kb.shape[0] == 1
        assert float(ks[0]) == pytest.approx(0.9)

    def test_apply_nms_suppresses_overlapping_boxes(self) -> None:
        # Two highly overlapping boxes of the same class
        boxes = torch.tensor([[0, 0, 100, 100], [5, 5, 95, 95]], dtype=torch.float32)
        labels = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.8])
        kb, kl, ks = apply_nms(boxes, labels, scores, iou_threshold=0.3)
        assert kb.shape[0] == 1

    def test_apply_nms_keeps_different_class_overlaps(self) -> None:
        boxes = torch.tensor([[0, 0, 100, 100], [5, 5, 95, 95]], dtype=torch.float32)
        labels = torch.tensor([0, 1])  # different classes
        scores = torch.tensor([0.9, 0.8])
        kb, kl, ks = apply_nms(boxes, labels, scores, iou_threshold=0.3)
        assert kb.shape[0] == 2  # both kept

    def test_apply_nms_empty_input(self) -> None:
        boxes = torch.zeros(0, 4)
        labels = torch.zeros(0, dtype=torch.long)
        scores = torch.zeros(0)
        kb, kl, ks = apply_nms(boxes, labels, scores)
        assert kb.shape[0] == 0

    def test_compute_box_iou_matrix_shape(self) -> None:
        boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
        iou = compute_box_iou_matrix(boxes)
        assert iou.shape == (2, 2)

    def test_compute_box_iou_self_is_one(self) -> None:
        boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        iou = compute_box_iou_matrix(boxes)
        assert float(iou[0, 0]) == pytest.approx(1.0)

    def test_compute_box_iou_no_overlap(self) -> None:
        boxes = torch.tensor([[0, 0, 5, 5], [10, 10, 20, 20]], dtype=torch.float32)
        iou = compute_box_iou_matrix(boxes)
        assert float(iou[0, 1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Inferencer tests
# ---------------------------------------------------------------------------


class TestDamageDetectionInferencer:
    def _make_inferencer(self) -> DamageDetectionInferencer:
        model = AnchorFreeDamageDetector(
            num_classes=NUM_CLASSES, class_names=DAMAGE_CLASS_NAMES, fpn_channels=32
        )
        cfg = DamageDetectionInferencerConfig(
            device="cpu", image_size=(64, 64), score_threshold=0.0
        )
        return DamageDetectionInferencer(model=model, config=cfg)

    def test_predict_from_array(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        pred = inferencer.predict_from_array(image)
        assert pred.num_detections >= 0

    def test_predict_from_path(self, tmp_path) -> None:
        from PIL import Image as _Image
        img = _Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(path)
        inferencer = self._make_inferencer()
        pred = inferencer.predict_from_path(path)
        assert hasattr(pred, "num_detections")

    def test_predict_batch(self) -> None:
        inferencer = self._make_inferencer()
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        preds = inferencer.predict_batch(images)
        assert len(preds) == 3

    def test_to_dict(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pred = inferencer.predict_from_array(image)
        d = pred.to_dict()
        assert "num_detections" in d
        assert "detections" in d
        assert isinstance(d["detections"], list)

    def test_visualize_returns_array(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pred = inferencer.predict_from_array(image)
        vis = pred.visualize(image)
        assert vis.shape == image.shape
        assert vis.dtype == np.uint8


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestDetectionUtils:
    def test_draw_detections_no_instances(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = draw_detections(image, [], class_names=DAMAGE_CLASS_NAMES)
        assert np.array_equal(result, image)

    def test_scale_boxes_to_original(self) -> None:
        boxes = [[0.0, 0.0, 64.0, 64.0]]
        scaled = scale_boxes_to_original(boxes, model_size=(64, 64), original_size=(128, 128))
        assert scaled[0] == pytest.approx([0.0, 0.0, 128.0, 128.0])


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


class TestDetectionMetrics:
    def test_map_perfect_match(self) -> None:
        preds = [{"boxes": [[0, 0, 10, 10]], "labels": [0], "scores": [0.9]}]
        gts = [{"boxes": [[0, 0, 10, 10]], "labels": [0]}]
        result = compute_map(preds, gts, iou_threshold=0.5, num_classes=1)
        assert result["map"] == pytest.approx(1.0, abs=0.01)

    def test_map_no_detections(self) -> None:
        preds = [{"boxes": [], "labels": [], "scores": []}]
        gts = [{"boxes": [[0, 0, 10, 10]], "labels": [0]}]
        result = compute_map(preds, gts, iou_threshold=0.5, num_classes=1)
        assert result["map"] == pytest.approx(0.0, abs=0.01)

    def test_map_empty_gt(self) -> None:
        preds = [{"boxes": [[0, 0, 10, 10]], "labels": [0], "scores": [0.9]}]
        gts = [{"boxes": [], "labels": []}]
        result = compute_map(preds, gts, iou_threshold=0.5, num_classes=1)
        # No GT → no AP computed → map = 0.0
        assert result["map"] == pytest.approx(0.0, abs=0.01)
