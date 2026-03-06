"""Full pipeline router."""

from __future__ import annotations

import io
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from backend.dependencies import get_pipeline
from backend.schemas.responses import (
    BoundingBox,
    PipelineAnalysisResponse,
    PipelineDamageInstance,
    PipelineMaterialSummary,
    PipelineSegmentationSummary,
)
from ml.pipeline.pipeline import BuildingAnalysisPipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _read_image(upload: UploadFile) -> np.ndarray:
    try:
        data = upload.file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc


@router.post(
    "/analyze",
    response_model=PipelineAnalysisResponse,
    summary="Full building facade analysis",
    description=(
        "Run the complete analysis pipeline on a building facade image:\n\n"
        "1. **Facade segmentation** — classifies every pixel into structural classes.\n"
        "2. **Damage detection** — locates and scores damage regions.\n"
        "3. **Material classification** — identifies materials in each region.\n\n"
        "Returns a structured JSON report of all stages."
    ),
)
async def analyze_facade(
    file: Annotated[UploadFile, File(description="Building facade image (JPEG/PNG).")],
    pipeline: BuildingAnalysisPipeline = Depends(get_pipeline),
) -> PipelineAnalysisResponse:
    """Analyze a building facade image through the full ML pipeline.

    Args:
        file: Uploaded image file.
        pipeline: Injected analysis pipeline.

    Returns:
        :class:`~backend.schemas.responses.PipelineAnalysisResponse`.
    """
    image = _read_image(file)
    result = pipeline.analyze(image)

    seg = result.segmentation
    fractions = seg.class_area_fractions
    dominant = max(fractions, key=fractions.get) if fractions else "unknown"

    damage_instances = []
    for di in result.damage_instances:
        x1, y1, x2, y2 = di.box
        damage_instances.append(
            PipelineDamageInstance(
                label=di.label,
                label_name=di.label_name,
                score=di.score,
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                material_in_region=di.material_in_region,
                material_score=di.material_score,
            )
        )

    return PipelineAnalysisResponse(
        image_path=result.image_path,
        segmentation=PipelineSegmentationSummary(
            class_area_fractions=fractions,
            dominant_class=dominant,
            damaged_area_fraction=fractions.get("damaged", 0.0),
        ),
        num_damage_instances=result.num_damage_instances,
        damage_instances=damage_instances,
        materials=PipelineMaterialSummary(
            overall_dominant_material=result.materials.overall_dominant_material,
            intact_material=result.materials.intact_material,
            damaged_material=result.materials.damaged_material,
        ),
        metadata=result.metadata,
    )
