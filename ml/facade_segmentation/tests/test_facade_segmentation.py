"""Unit and integration tests for the facade segmentation module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.common.base_model import SegmentationResult
from ml.common.metrics import compute_iou, compute_pixel_accuracy
from ml.common.registry import ModelRegistry
from ml.common.transforms import get_segmentation_transforms
from ml.facade_segmentation.dataset import FACADE_CLASS_NAMES
from ml.facade_segmentation.inference import SegmentationInferencer, SegmentationInferencerConfig
from ml.facade_segmentation.model import DeepLabV3PlusSegmentation, UNetSegmentation
from ml.facade_segmentation.utils import (
    colorize_mask,
    compute_class_statistics,
    extract_class_masks,
    overlay_mask_on_image,
)


NUM_CLASSES = len(FACADE_CLASS_NAMES)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestDeepLabV3PlusSegmentation:
    def test_output_shape(self) -> None:
        model = DeepLabV3PlusSegmentation(num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES, 64, 64), f"Unexpected shape: {out.shape}"

    def test_predict_returns_results(self) -> None:
        model = DeepLabV3PlusSegmentation(num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES)
        x = torch.randn(1, 3, 64, 64)
        results = model.predict(x)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SegmentationResult)
        assert r.mask.shape == (64, 64)
        assert r.probabilities.shape == (NUM_CLASSES, 64, 64)
        assert r.logits.shape == (NUM_CLASSES, 64, 64)

    def test_predict_mask_values_in_range(self) -> None:
        model = DeepLabV3PlusSegmentation(num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES)
        x = torch.randn(1, 3, 32, 32)
        results = model.predict(x)
        mask = results[0].mask
        assert mask.min() >= 0
        assert mask.max() < NUM_CLASSES

    def test_save_and_load(self, tmp_path) -> None:
        model = DeepLabV3PlusSegmentation(num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES)
        path = tmp_path / "seg_model.pt"
        model.save(path)
        model2 = DeepLabV3PlusSegmentation(num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES)
        model2.load(path)
        # Both models should produce identical outputs
        x = torch.randn(1, 3, 32, 32)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)

    def test_wrong_class_names_raises(self) -> None:
        with pytest.raises(ValueError, match="class_names length"):
            DeepLabV3PlusSegmentation(num_classes=3, class_names=["a", "b"])  # mismatch


class TestUNetSegmentation:
    def test_output_shape(self) -> None:
        model = UNetSegmentation(num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES, base_channels=8)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES, 64, 64)

    def test_predict(self) -> None:
        model = UNetSegmentation(num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES, base_channels=8)
        x = torch.randn(1, 3, 64, 64)
        results = model.predict(x)
        assert len(results) == 1
        assert results[0].mask.shape == (64, 64)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_deeplabv3plus_registered(self) -> None:
        names = ModelRegistry.list_models(namespace="segmentation")
        assert "deeplabv3plus" in names

    def test_unet_registered(self) -> None:
        names = ModelRegistry.list_models(namespace="segmentation")
        assert "unet" in names

    def test_build_from_registry(self) -> None:
        model = ModelRegistry.build(
            "unet",
            namespace="segmentation",
            num_classes=NUM_CLASSES,
            class_names=FACADE_CLASS_NAMES,
            base_channels=8,
        )
        assert isinstance(model, UNetSegmentation)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            ModelRegistry.build("nonexistent_model", namespace="segmentation")


# ---------------------------------------------------------------------------
# Inferencer tests
# ---------------------------------------------------------------------------


class TestSegmentationInferencer:
    def _make_inferencer(self) -> SegmentationInferencer:
        model = UNetSegmentation(
            num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES, base_channels=8
        )
        cfg = SegmentationInferencerConfig(device="cpu", image_size=(64, 64))
        return SegmentationInferencer(model=model, config=cfg)

    def test_predict_from_array(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        pred = inferencer.predict_from_array(image)
        assert pred.mask.shape == (64, 64)
        assert pred.probabilities.shape == (NUM_CLASSES, 64, 64)

    def test_predict_from_path(self, tmp_path) -> None:
        from PIL import Image as _Image
        img = _Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        inferencer = self._make_inferencer()
        pred = inferencer.predict_from_path(img_path)
        assert pred.mask.shape == (64, 64)

    def test_predict_batch(self) -> None:
        inferencer = self._make_inferencer()
        images = [np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8) for _ in range(3)]
        preds = inferencer.predict_batch(images)
        assert len(preds) == 3
        for p in preds:
            assert p.mask.shape == (64, 64)

    def test_class_area_fractions_sums_to_one(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pred = inferencer.predict_from_array(image)
        fractions = pred.class_area_fractions()
        total = sum(fractions.values())
        assert abs(total - 1.0) < 1e-5

    def test_colored_mask_is_pil_image(self) -> None:
        inferencer = self._make_inferencer()
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pred = inferencer.predict_from_array(image)
        from PIL import Image as _Image
        assert isinstance(pred.colored_mask, _Image.Image)

    def test_tta(self) -> None:
        model = UNetSegmentation(
            num_classes=NUM_CLASSES, class_names=FACADE_CLASS_NAMES, base_channels=8
        )
        cfg = SegmentationInferencerConfig(device="cpu", image_size=(64, 64), tta=True)
        inferencer = SegmentationInferencer(model=model, config=cfg)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pred = inferencer.predict_from_array(image)
        assert pred.mask.shape == (64, 64)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestUtils:
    def test_colorize_mask_shape(self) -> None:
        mask = np.zeros((32, 32), dtype=np.int64)
        img = colorize_mask(mask)
        assert img.size == (32, 32)

    def test_overlay_mask_on_image(self) -> None:
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.random.randint(0, NUM_CLASSES, (64, 64), dtype=np.int64)
        result = overlay_mask_on_image(image, mask, alpha=0.5)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_overlay_resizes_mask(self) -> None:
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.randint(0, NUM_CLASSES, (64, 64), dtype=np.int64)
        result = overlay_mask_on_image(image, mask)
        assert result.shape == (128, 128, 3)

    def test_extract_class_masks(self) -> None:
        mask = np.array([[0, 1], [2, 0]], dtype=np.int64)
        class_masks = extract_class_masks(mask)
        assert "background" in class_masks
        assert "wall" in class_masks
        assert class_masks["background"].shape == (2, 2)

    def test_compute_class_statistics(self) -> None:
        mask = np.zeros((10, 10), dtype=np.int64)
        mask[:5] = 1  # half of pixels are class 1
        stats = compute_class_statistics(mask)
        assert "background" in stats
        assert abs(stats["background"]["area_fraction"] - 0.5) < 1e-5
        assert abs(stats["wall"]["area_fraction"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Metrics tests (shared module)
# ---------------------------------------------------------------------------


class TestSegmentationMetrics:
    def test_iou_perfect(self) -> None:
        mask = torch.randint(0, NUM_CLASSES, (64, 64))
        result = compute_iou(mask, mask, num_classes=NUM_CLASSES)
        assert result["mean_iou"] == pytest.approx(1.0, abs=1e-5)

    def test_iou_all_wrong(self) -> None:
        pred = torch.zeros(64, 64, dtype=torch.long)
        gt = torch.ones(64, 64, dtype=torch.long)
        result = compute_iou(pred, gt, num_classes=NUM_CLASSES)
        assert result["mean_iou"] == pytest.approx(0.0, abs=1e-5)

    def test_pixel_accuracy_perfect(self) -> None:
        mask = torch.randint(0, NUM_CLASSES, (64, 64))
        acc = compute_pixel_accuracy(mask, mask)
        assert acc == pytest.approx(1.0, abs=1e-5)

    def test_pixel_accuracy_all_wrong(self) -> None:
        pred = torch.zeros(64, 64, dtype=torch.long)
        gt = torch.ones(64, 64, dtype=torch.long)
        acc = compute_pixel_accuracy(pred, gt)
        assert acc == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_segmentation_transform_shapes(self) -> None:
        transform = get_segmentation_transforms(image_size=(128, 128), is_train=False)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, NUM_CLASSES, (256, 256), dtype=np.int64)
        out = transform(image=image, mask=mask)
        assert out["image"].shape == (3, 128, 128)
        assert out["mask"].shape == (128, 128)

    def test_train_transform_returns_tensor(self) -> None:
        transform = get_segmentation_transforms(image_size=(64, 64), is_train=True)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.int64)
        out = transform(image=image, mask=mask)
        assert isinstance(out["image"], torch.Tensor)
