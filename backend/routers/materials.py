"""Material classification router."""

from __future__ import annotations

import io
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image

from backend.dependencies import get_material_inferencer
from backend.schemas.responses import (
    BoundingBox,
    MaterialClassificationResponse,
    RegionMaterialResponse,
    RegionMaterialResult,
)
from ml.material_classification.inference import MaterialInferencer

router = APIRouter(prefix="/materials", tags=["materials"])


def _read_image(upload: UploadFile) -> np.ndarray:
    try:
        data = upload.file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc


@router.post(
    "/predict",
    response_model=MaterialClassificationResponse,
    summary="Material classification (whole image)",
    description=(
        "Classify the dominant building material in the uploaded image or crop. "
        "Returns the predicted material class and per-class confidence scores."
    ),
)
async def predict_material(
    file: Annotated[UploadFile, File(description="Building facade image or region crop.")],
    inferencer: MaterialInferencer = Depends(get_material_inferencer),
) -> MaterialClassificationResponse:
    """Classify the material in the uploaded image.

    Args:
        file: Uploaded image file.
        inferencer: Injected material classification inferencer.

    Returns:
        :class:`~backend.schemas.responses.MaterialClassificationResponse`.
    """
    image = _read_image(file)
    result = inferencer.predict_from_array(image)
    return MaterialClassificationResponse(
        label=result.label,
        label_name=result.label_name,
        score=result.score,
        scores=result.scores,
    )


@router.post(
    "/predict-regions",
    response_model=RegionMaterialResponse,
    summary="Material classification (bounding-box regions)",
    description=(
        "Classify building materials within specific bounding-box regions of the "
        "uploaded image.  Boxes are provided as query parameters: "
        "``boxes=x1,y1,x2,y2&boxes=x1,y1,x2,y2&...``."
    ),
)
async def predict_material_regions(
    file: Annotated[UploadFile, File(description="Building facade image.")],
    boxes: Annotated[
        list[str],
        Query(description="Bounding boxes as comma-separated x1,y1,x2,y2 strings."),
    ] = [],
    inferencer: MaterialInferencer = Depends(get_material_inferencer),
) -> RegionMaterialResponse:
    """Classify materials within bounding-box regions.

    Args:
        file: Uploaded image file.
        boxes: List of box strings in ``"x1,y1,x2,y2"`` format (query params).
        inferencer: Injected material classification inferencer.

    Returns:
        :class:`~backend.schemas.responses.RegionMaterialResponse`.
    """
    image = _read_image(file)

    parsed_boxes: list[list[float]] = []
    for box_str in boxes:
        try:
            coords = [float(v) for v in box_str.split(",")]
            if len(coords) != 4:
                raise ValueError("Expected 4 values")
            parsed_boxes.append(coords)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid box format '{box_str}': {exc}",
            ) from exc

    region_results = inferencer.predict_regions(image, parsed_boxes)

    results = []
    for r in region_results:
        x1, y1, x2, y2 = r["box"]
        results.append(
            RegionMaterialResult(
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                label=r["label"],
                label_name=r["label_name"],
                score=r["score"],
                scores=r["scores"],
            )
        )

    return RegionMaterialResponse(results=results)
