"""POST /api/analyze — run ML pipeline, persist result, return response skeleton."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend.api import deps
from backend.api.schemas import (
    AnalysisResponse,
    DamageItem,
    LayerAnalysisItem,
    MaskUrls,
    MaterialItem,
)
from backend.calibration import ScaleCalibration, fallback_from_total_area
from backend.core.config import get_settings
from backend.core.results_store import results_store
from backend.estimator.calculator import build_estimate_from_analysis

router = APIRouter()
logger = logging.getLogger(__name__)


def _mask_to_png_path(mask: np.ndarray, out_dir: Path, name: str) -> str:
    """Save a boolean mask as an 8-bit PNG (0/255) for Flutter overlay."""
    import cv2

    out = (mask.astype(np.uint8) * 255)
    path = out_dir / f"mask_{name}.png"
    cv2.imwrite(str(path), out, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return str(path)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    calibration_id: Optional[str] = Form(default=None),
    total_area_m2: Optional[float] = Form(default=None),
    analyzer=Depends(deps.get_analyzer),
):
    settings = get_settings()

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.max_image_bytes // 1_000_000}MB)",
        )

    logger.info("Analyze: %s (%d bytes)", file.filename, len(image_bytes))
    result = analyzer.analyze(image_bytes, output_dir=str(settings.results_dir))

    analysis_id = result["id"]
    result_dir = Path(settings.results_dir) / analysis_id

    # Resolve scale calibration
    calibration_warnings: list[str] = []
    cal: Optional[ScaleCalibration] = None
    if calibration_id:
        payload = await results_store.get_calibration(calibration_id)
        if payload:
            cal = ScaleCalibration(
                px_per_m_linear=payload["px_per_m_linear"],
                m2_per_px=payload["m2_per_px"],
                reference_type=payload.get("reference_type", "custom"),
                reference_width_m=payload.get("reference_width_m", 0.0),
                reference_height_m=payload.get("reference_height_m"),
                warnings=payload.get("warnings", []),
            )
            calibration_warnings = list(cal.warnings)

    if cal is None:
        fallback_area = total_area_m2 or 450.0
        cal = fallback_from_total_area(fallback_area, result["total_area_px"] or 1)
        calibration_warnings = list(cal.warnings)

    # Save per-class masks as PNG so Flutter can overlay them
    masks_dir = result_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    geom_mask_files: dict[str, str] = {}
    defect_mask_files: dict[str, str] = {}
    material_mask_files: dict[str, str] = {}

    for name, mask in (result.get("masks", {}).get("geometry") or {}).items():
        if mask is None or not np.any(mask):
            continue
        geom_mask_files[name] = _mask_to_png_path(mask, masks_dir, f"geom_{name}")

    for name, mask in (result.get("masks", {}).get("defects") or {}).items():
        if mask is None or not np.any(mask):
            continue
        defect_mask_files[name] = _mask_to_png_path(mask, masks_dir, f"defect_{name}")

    for name, mask in (result.get("masks", {}).get("materials") or {}).items():
        if mask is None or not np.any(mask):
            continue
        material_mask_files[name] = _mask_to_png_path(mask, masks_dir, f"material_{name}")

    # Register images in the store (viz + base + masks)
    image_paths: dict[str, str] = {"original": str(result_dir / "original.jpg")}
    image_paths.update({f"viz_{k}": v for k, v in result.get("image_paths", {}).items()})
    image_paths.update({f"geom_{k}": v for k, v in geom_mask_files.items()})
    image_paths.update({f"defect_{k}": v for k, v in defect_mask_files.items()})
    image_paths.update({f"material_{k}": v for k, v in material_mask_files.items()})

    # Convert areas to real metres
    damages_out: list[DamageItem] = []
    for d in result["damages"]:
        d_area_m2 = cal.area_px_to_m2(d["area_px"])
        damages_out.append(
            DamageItem(
                type=d["type"],
                type_display=d["type_display"],
                percentage=d["percentage"],
                area_px=d["area_px"],
                area_m2=round(d_area_m2, 2),
                severity=d["severity"],
                severity_display=d["severity_display"],
                affected_layers=d.get("affected_layers", []),
                crack_depth=d.get("crack_depth"),
            )
        )

    materials_out: list[MaterialItem] = []
    for m in result["materials"]:
        m_area_m2 = cal.area_px_to_m2(m["area_px"])
        materials_out.append(
            MaterialItem(
                name=m["name"],
                name_display=m["name_display"],
                percentage=m["percentage"],
                area_px=m["area_px"],
                area_m2=round(m_area_m2, 2),
            )
        )

    layer_analysis_out: dict[str, LayerAnalysisItem] = {}
    for k, v in (result.get("layer_analysis") or {}).items():
        layer_analysis_out[k] = LayerAnalysisItem(
            area_px=v["area_px"],
            area_m2=round(cal.area_px_to_m2(v["area_px"]), 2),
            affected_layers=v["affected_layers"],
            crack_depth=v.get("crack_depth"),
        )

    damaged_area_m2 = round(cal.area_px_to_m2(result["damaged_area_px"]), 2)
    total_area_m2_out = round(cal.area_px_to_m2(result["total_area_px"]), 2)

    # Build URLs that the client can fetch directly.
    base_url = str(request.base_url).rstrip("/")
    prefix = settings.api_prefix

    def url_for(key: str) -> str:
        return f"{base_url}{prefix}/results/{analysis_id}/image/{key}"

    masks = MaskUrls(
        base_image=url_for("original"),
        geometry={k: url_for(f"geom_{k}") for k in geom_mask_files},
        materials={k: url_for(f"material_{k}") for k in material_mask_files},
        defects={k: url_for(f"defect_{k}") for k in defect_mask_files},
        visualizations={k: url_for(f"viz_{k}") for k in result.get("image_paths", {})},
    )

    # Build repair estimate in RUB (uses live or YAML-fallback prices)
    estimate_result = await build_estimate_from_analysis(
        damages=[d.model_dump() for d in damages_out],
        layer_analysis={k: v.model_dump() for k, v in layer_analysis_out.items()},
        scale=cal,
    )

    response = AnalysisResponse(
        id=analysis_id,
        overall_score=result["overall_score"],
        overall_condition=result["overall_condition"],
        total_area_px=result["total_area_px"],
        total_area_m2=total_area_m2_out,
        damaged_area_px=result["damaged_area_px"],
        damaged_area_m2=damaged_area_m2,
        calibration_id=calibration_id,
        calibration_warnings=calibration_warnings,
        damages=damages_out,
        materials=materials_out,
        layer_analysis=layer_analysis_out,
        processed_images=result.get("processed_images", []),
        masks=masks,
        price_snapshot_date=estimate_result.get("price_snapshot_date"),
        price_source=estimate_result.get("price_source"),
        repair_estimate=estimate_result.get("repair_estimate", {}),
    )

    await results_store.put_analysis(
        analysis_id, response.model_dump(mode="json"), image_paths
    )
    return response
