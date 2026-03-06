"""Image augmentation and pre-processing transform factories.

Transforms are built with `albumentations` for its rich augmentation library
and native support for masks and bounding boxes.  Each factory function returns
a composed pipeline suitable for a particular task.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_segmentation_transforms(
    image_size: tuple[int, int] = (512, 512),
    *,
    is_train: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Return an albumentations pipeline for semantic segmentation.

    The same spatial transforms are applied consistently to both the image and
    its segmentation mask.

    Args:
        image_size: Target ``(height, width)`` after resizing.
        is_train: Whether to include data augmentation.
        mean: Per-channel normalisation mean (ImageNet defaults).
        std: Per-channel normalisation standard deviation.

    Returns:
        An :class:`albumentations.Compose` transform that accepts
        ``image`` (H×W×3 uint8) and ``mask`` (H×W int) keyword arguments.
    """
    h, w = image_size
    common = [
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    if not is_train:
        return A.Compose(common)

    train_augs = [
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(train_augs)


def get_detection_transforms(
    image_size: tuple[int, int] = (640, 640),
    *,
    is_train: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    bbox_format: str = "pascal_voc",
) -> A.Compose:
    """Return an albumentations pipeline for object detection.

    Spatial augmentations are applied to *both* the image and its bounding
    boxes (and optional instance masks).

    Args:
        image_size: Target ``(height, width)`` after resizing.
        is_train: Whether to include data augmentation.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation standard deviation.
        bbox_format: Format for bounding boxes passed to the transform.
            One of ``"pascal_voc"`` (x1,y1,x2,y2), ``"coco"``
            (x,y,w,h), ``"yolo"`` (cx,cy,w,h normalised).

    Returns:
        An :class:`albumentations.Compose` configured for bounding-box
        augmentation.
    """
    h, w = image_size
    bbox_params = A.BboxParams(
        format=bbox_format,
        label_fields=["class_labels"],
        min_visibility=0.3,
    )

    common = [
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    if not is_train:
        return A.Compose(common, bbox_params=bbox_params)

    train_augs = [
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(train_augs, bbox_params=bbox_params)


def get_classification_transforms(
    image_size: tuple[int, int] = (224, 224),
    *,
    is_train: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Return an albumentations pipeline for image / region classification.

    Args:
        image_size: Target ``(height, width)`` after resizing.
        is_train: Whether to include data augmentation.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation standard deviation.

    Returns:
        An :class:`albumentations.Compose` transform that accepts an
        ``image`` (H×W×3 uint8) keyword argument.
    """
    h, w = image_size

    common = [
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    if not is_train:
        return A.Compose(common)

    train_augs = [
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-30, 30), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(train_augs)
