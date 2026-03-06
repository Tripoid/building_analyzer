"""Damage detection router."""

from __future__ import annotations

import io
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image

from backend.dependencies import get_damage_inferencer
from backend.schemas.responses import BoundingBox, DamageDetectionInstance, DamageDetectionResponse
from ml.damage_detection.inference import DamageDetectionInferencer

router = APIRouter(prefix="/damage", tags=["damage"])


def _read_image(upload: UploadFile) -> np.ndarray:
    try:
        data = upload.file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc


@router.post(
    "/predict",
    response_model=DamageDetectionResponse,
    summary="Damage detection",
    description=(
        "Detect and highlight damaged areas in a building facade image. "
        "Returns bounding boxes and confidence scores for each damage instance."
    ),
)
async def predict_damage(
    file: Annotated[UploadFile, File(description="Building facade image (JPEG/PNG).")],
    score_threshold: float = Query(
        default=0.4, ge=0.0, le=1.0, description="Minimum confidence threshold."
    ),
    inferencer: DamageDetectionInferencer = Depends(get_damage_inferencer),
) -> DamageDetectionResponse:
    """Detect damage regions in the uploaded facade image.

    Args:
        file: Uploaded image file.
        score_threshold: Override the model's default score threshold.
        inferencer: Injected damage detection inferencer.

    Returns:
        :class:`~backend.schemas.responses.DamageDetectionResponse`.
    """
    image = _read_image(file)

    # Temporarily override the score threshold for this request
    original_threshold = inferencer.cfg.score_threshold
    inferencer.cfg.score_threshold = score_threshold
    try:
        prediction = inferencer.predict_from_array(image)
    finally:
        inferencer.cfg.score_threshold = original_threshold

    detections = []
    for inst in prediction.result.instances:
        x1, y1, x2, y2 = inst.box
        label_name = (
            inferencer.model.class_names[inst.label]
            if inst.label < len(inferencer.model.class_names)
            else "unknown"
        )
        detections.append(
            DamageDetectionInstance(
                label=inst.label,
                label_name=label_name,
                score=inst.score,
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )
        )

    return DamageDetectionResponse(
        num_detections=len(detections),
        detections=detections,
    )
