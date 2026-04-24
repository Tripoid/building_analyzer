"""POST /api/calibrate — compute pixel↔metre scale for a photo."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from backend.api.schemas import CalibrationRequest, CalibrationResponse
from backend.calibration import (
    calibrate_from_bbox,
    calibrate_from_points,
)
from backend.core.results_store import results_store

router = APIRouter()


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate(req: CalibrationRequest) -> CalibrationResponse:
    try:
        if req.p1 is not None and req.p2 is not None:
            cal = calibrate_from_points(
                p1=req.p1,
                p2=req.p2,
                reference_width_m=req.reference_width_m,
                reference_type=req.reference_type,
                reference_height_m=req.reference_height_m,
            )
        else:
            assert req.bbox is not None
            cal = calibrate_from_bbox(
                bbox_xyxy=req.bbox,
                reference_width_m=req.reference_width_m,
                reference_type=req.reference_type,
                reference_height_m=req.reference_height_m,
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    calibration_id = str(uuid.uuid4())
    payload = {
        **cal.to_dict(),
        "image_width_px": req.image_width_px,
        "image_height_px": req.image_height_px,
    }
    await results_store.put_calibration(calibration_id, payload)

    return CalibrationResponse(
        calibration_id=calibration_id,
        px_per_m=cal.px_per_m_linear,
        m2_per_px=cal.m2_per_px,
        reference_type=cal.reference_type,
        reference_width_m=cal.reference_width_m,
        reference_height_m=cal.reference_height_m,
        warnings=cal.warnings,
    )
