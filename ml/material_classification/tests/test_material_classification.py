"""Unit and integration tests for the material classification module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.common.base_model import ClassificationResult
from ml.common.metrics import compute_top_k_accuracy
from ml.common.registry import ModelRegistry
from ml.common.transforms import get_classification_transforms
from ml.material_classification.dataset import MATERIAL_CLASS_NAMES
from ml.material_classification.inference import MaterialInferencer, MaterialInferencerConfig
from ml.material_classification.model import CNNMaterialClassifier, ViTMaterialClassifier
from ml.material_classification.utils import (
    aggregate_region_materials,
    extract_region_crops,
)

NUM_CLASSES = len(MATERIAL_CLASS_NAMES)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestCNNMaterialClassifier:
    def _make_model(self, base_channels: int = 8) -> CNNMaterialClassifier:
        return CNNMaterialClassifier(
            num_classes=NUM_CLASSES,
            class_names=MATERIAL_CLASS_NAMES,
            base_channels=base_channels,
        )

    def test_output_shape(self) -> None:
        model = self._make_model()
        model.eval()
        x = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_predict_returns_results(self) -> None:
        model = self._make_model()
        x = torch.randn(2, 3, 64, 64)
        results = model.predict(x)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ClassificationResult)
            assert 0 <= r.label < NUM_CLASSES
            assert 0.0 <= r.score <= 1.0

    def test_predict_scores_sum_to_one(self) -> None:
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64)
        results = model.predict(x)
        total = sum(results[0].scores.values())
        assert abs(total - 1.0) < 1e-5

    def test_predict_label_name_correct(self) -> None:
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64)
        r = model.predict(x)[0]
        assert r.label_name == MATERIAL_CLASS_NAMES[r.label]

    def test_save_and_load(self, tmp_path) -> None:
        model = self._make_model()
        path = tmp_path / "cnn_material.pt"
        model.save(path)
        model2 = self._make_model()
        model2.load(path)
        x = torch.randn(1, 3, 32, 32)
        model.eval()
        model2.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), model2(x))

    def test_wrong_num_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="class_names length"):
            CNNMaterialClassifier(num_classes=3, class_names=["a"])


class TestViTMaterialClassifier:
    def _make_model(self) -> ViTMaterialClassifier:
        return ViTMaterialClassifier(
            num_classes=NUM_CLASSES,
            class_names=MATERIAL_CLASS_NAMES,
            img_size=32,
            patch_size=8,
            embed_dim=32,
            depth=2,
            num_heads=2,
        )

    def test_output_shape(self) -> None:
        model = self._make_model()
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_predict_returns_results(self) -> None:
        model = self._make_model()
        x = torch.randn(1, 3, 32, 32)
        results = model.predict(x)
        assert len(results) == 1
        assert isinstance(results[0], ClassificationResult)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestClassificationRegistry:
    def test_cnn_registered(self) -> None:
        assert "cnn_classifier" in ModelRegistry.list_models(namespace="classification")

    def test_vit_registered(self) -> None:
        assert "vit_classifier" in ModelRegistry.list_models(namespace="classification")

    def test_build_cnn(self) -> None:
        model = ModelRegistry.build(
            "cnn_classifier",
            namespace="classification",
            num_classes=NUM_CLASSES,
            class_names=MATERIAL_CLASS_NAMES,
            base_channels=8,
        )
        assert isinstance(model, CNNMaterialClassifier)


# ---------------------------------------------------------------------------
# Inferencer tests
# ---------------------------------------------------------------------------


class TestMaterialInferencer:
    def _make_inferencer(self) -> MaterialInferencer:
        model = CNNMaterialClassifier(
            num_classes=NUM_CLASSES, class_names=MATERIAL_CLASS_NAMES, base_channels=8
        )
        cfg = MaterialInferencerConfig(device="cpu", image_size=(32, 32), batch_size=4)
        return MaterialInferencer(model=model, config=cfg)

    def test_predict_from_array(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = inferencer.predict_from_array(image)
        assert isinstance(result, ClassificationResult)

    def test_predict_from_path(self, tmp_path) -> None:
        from PIL import Image as _Image
        img = _Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        path = tmp_path / "material.jpg"
        img.save(path)
        inferencer = self._make_inferencer()
        result = inferencer.predict_from_path(path)
        assert isinstance(result, ClassificationResult)

    def test_predict_batch_size(self) -> None:
        inferencer = self._make_inferencer()
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        results = inferencer.predict_batch(images)
        assert len(results) == 5

    def test_predict_regions(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        boxes = [[10, 10, 100, 100], [50, 50, 150, 150]]
        results = inferencer.predict_regions(image, boxes)
        assert len(results) == 2
        for r in results:
            assert "label_name" in r
            assert "score" in r
            assert "box" in r

    def test_predict_regions_empty_boxes(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        results = inferencer.predict_regions(image, [])
        assert results == []

    def test_predict_mask_regions(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[10:30, 10:30] = 1
        results = inferencer.predict_mask_regions(image, mask)
        assert 0 in results or 1 in results  # at least one class classified


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestMaterialUtils:
    def test_extract_region_crops_shapes(self) -> None:
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        boxes = [[10, 10, 50, 80], [100, 100, 180, 180]]
        crops = extract_region_crops(image, boxes)
        assert len(crops) == 2
        assert crops[0].shape == (70, 40, 3)
        assert crops[1].shape == (80, 80, 3)

    def test_extract_region_crops_with_target_size(self) -> None:
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        boxes = [[0, 0, 100, 100]]
        crops = extract_region_crops(image, boxes, target_size=(32, 32))
        assert crops[0].shape == (32, 32, 3)

    def test_extract_region_crops_degenerate_box(self) -> None:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        boxes = [[50, 50, 50, 50]]  # zero area
        crops = extract_region_crops(image, boxes)
        assert len(crops) == 1  # should not raise

    def test_aggregate_region_materials_empty(self) -> None:
        result = aggregate_region_materials([])
        assert result["dominant_material"] is None
        assert result["material_counts"] == {}

    def test_aggregate_region_materials_dominant(self) -> None:
        regions = [
            {"label_name": "concrete", "score": 0.9, "scores": {"concrete": 0.9}},
            {"label_name": "concrete", "score": 0.8, "scores": {"concrete": 0.8}},
            {"label_name": "brick", "score": 0.7, "scores": {"brick": 0.7}},
        ]
        result = aggregate_region_materials(regions)
        assert result["dominant_material"] == "concrete"
        assert result["material_counts"]["concrete"] == 2
        assert result["material_counts"]["brick"] == 1


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_top1_perfect(self) -> None:
        logits = torch.eye(NUM_CLASSES)
        targets = torch.arange(NUM_CLASSES)
        acc = compute_top_k_accuracy(logits, targets, k=1)
        assert acc == pytest.approx(1.0, abs=1e-5)

    def test_top1_all_wrong(self) -> None:
        logits = torch.zeros(4, NUM_CLASSES)
        logits[:, 0] = 10.0  # always predict class 0
        targets = torch.ones(4, dtype=torch.long)  # but GT is class 1
        acc = compute_top_k_accuracy(logits, targets, k=1)
        assert acc == pytest.approx(0.0, abs=1e-5)

    def test_top5_higher_than_top1(self) -> None:
        logits = torch.randn(32, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (32,))
        top1 = compute_top_k_accuracy(logits, targets, k=1)
        top5 = compute_top_k_accuracy(logits, targets, k=min(5, NUM_CLASSES))
        assert top5 >= top1

    def test_empty_logits(self) -> None:
        logits = torch.zeros(0, NUM_CLASSES)
        targets = torch.zeros(0, dtype=torch.long)
        acc = compute_top_k_accuracy(logits, targets, k=1)
        assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestClassificationTransforms:
    def test_output_shape(self) -> None:
        transform = get_classification_transforms(image_size=(32, 32), is_train=False)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = transform(image=image)
        assert out["image"].shape == (3, 32, 32)

    def test_train_transform(self) -> None:
        transform = get_classification_transforms(image_size=(64, 64), is_train=True)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = transform(image=image)
        assert isinstance(out["image"], torch.Tensor)
        assert out["image"].shape == (3, 64, 64)
