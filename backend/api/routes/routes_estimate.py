"""POST /api/estimate — rebuild a repair estimate for an existing analysis."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas import EstimateRequest, EstimateResponse
from backend.calibration import ScaleCalibration
from backend.core.results_store import results_store
from backend.estimator.calculator import build_estimate_from_analysis

router = APIRouter()


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(req: EstimateRequest):
    rec = await results_store.get_analysis(req.analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    payload = rec.payload

    # Rebuild the ScaleCalibration from the saved per-px conversion.
    total_area_px = int(payload.get("total_area_px", 0) or 0)
    total_area_m2 = float(payload.get("total_area_m2", 0) or 0)
    m2_per_px = total_area_m2 / total_area_px if total_area_px > 0 else 0
    px_per_m = (m2_per_px ** -0.5) if m2_per_px > 0 else 0
    scale = ScaleCalibration(
        px_per_m_linear=px_per_m,
        m2_per_px=m2_per_px,
        reference_type="custom",
        reference_width_m=0.0,
        reference_height_m=None,
    )

    result = await build_estimate_from_analysis(
        damages=payload.get("damages", []),
        layer_analysis=payload.get("layer_analysis", {}),
        scale=scale,
        waste_factor=req.waste_factor,
        vat_rate=req.vat_rate,
    )

    # Persist the new estimate on the analysis payload for the Flutter client.
    payload["repair_estimate"] = result["repair_estimate"]
    payload["price_snapshot_date"] = result.get("price_snapshot_date")
    payload["price_source"] = result.get("price_source")
    await results_store.put_analysis(req.analysis_id, payload, rec.image_paths)

    return EstimateResponse(
        analysis_id=req.analysis_id,
        repair_estimate=result["repair_estimate"],
        price_snapshot_date=result.get("price_snapshot_date"),
        price_source=result.get("price_source", "yaml_fallback"),
        stale=result.get("stale", False),
    )
