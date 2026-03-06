"""Facade segmentation router."""

from __future__ import annotations

import base64
import io
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from backend.dependencies import get_segmentation_inferencer
from backend.schemas.responses import SegmentationResponse
from ml.facade_segmentation.inference import SegmentationInferencer

router = APIRouter(prefix="/segmentation", tags=["segmentation"])


def _read_image(upload: UploadFile) -> np.ndarray:
    """Read an uploaded file and return an RGB numpy array."""
    try:
        data = upload.file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc


@router.post(
    "/predict",
    response_model=SegmentationResponse,
    summary="Facade segmentation",
    description=(
        "Segment a building facade image into structural element classes: "
        "background, wall, window, door, balcony, cornice, damaged. "
        "Returns per-class area fractions and an optional coloured mask."
    ),
)
async def predict_segmentation(
    file: Annotated[UploadFile, File(description="Building facade image (JPEG/PNG).")],
    include_mask: bool = False,
    inferencer: SegmentationInferencer = Depends(get_segmentation_inferencer),
) -> SegmentationResponse:
    """Run facade segmentation on the uploaded image.

    Args:
        file: Uploaded image file.
        include_mask: If ``true``, include a base-64 encoded coloured mask PNG
            in the response.
        inferencer: Injected segmentation inferencer.

    Returns:
        :class:`~backend.schemas.responses.SegmentationResponse`.
    """
    image = _read_image(file)
    prediction = inferencer.predict_from_array(image)

    colored_b64: str | None = None
    if include_mask:
        buf = io.BytesIO()
        prediction.colored_mask.save(buf, format="PNG")
        colored_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    fractions = prediction.class_area_fractions()
    dominant = max(fractions, key=fractions.get) if fractions else "unknown"

    return SegmentationResponse(
        class_area_fractions=fractions,
        dominant_class=dominant,
        damaged_area_fraction=fractions.get("damaged", 0.0),
        colored_mask_b64=colored_b64,
    )
