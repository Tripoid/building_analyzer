"""POST /api/restore/{analysis_id} — AI inpainting of the detected defects."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from backend.api import deps
from backend.api.schemas import RestoreRequest, RestoreResponse
from backend.core.config import get_settings
from backend.core.results_store import results_store
from backend.restoration.mask_prep import prepare_restoration_mask

router = APIRouter()
logger = logging.getLogger(__name__)


def _load_defect_masks(analysis: dict, image_paths: dict[str, str]) -> dict[str, np.ndarray]:
    """Reload defect masks that /analyze stored as PNGs on disk."""
    masks: dict[str, np.ndarray] = {}
    for key, path in image_paths.items():
        if not key.startswith("defect_"):
            continue
        name = key[len("defect_"):]
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        masks[name] = arr > 127
    return masks


@router.post("/restore/{analysis_id}", response_model=RestoreResponse)
async def restore(analysis_id: str, req: RestoreRequest):
    rec = await results_store.get_analysis(analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    settings = get_settings()
    base_path = rec.image_paths.get("original")
    if not base_path or not Path(base_path).exists():
        raise HTTPException(status_code=404, detail="Base image missing on disk")

    defect_masks = _load_defect_masks(rec.payload, rec.image_paths)
    if not defect_masks:
        raise HTTPException(status_code=422, detail="No defect masks to restore")

    image_bgr = cv2.imread(base_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prep_mask = prepare_restoration_mask(defect_masks)

    # Pick provider: override via query param, otherwise settings default.
    provider_name = "sd" if req.quality == "high" else settings.inpaint_provider
    try:
        if provider_name == "sd":
            from backend.restoration.providers.sd_inpaint import SDInpaintProvider

            provider = SDInpaintProvider()
        else:
            from backend.restoration.providers.lama import LamaProvider

            provider = LamaProvider()
    except Exception as e:
        logger.exception("Inpaint provider init failed")
        raise HTTPException(status_code=500, detail=f"Provider init failed: {e}")

    start = time.perf_counter()
    try:
        restored_rgb = await provider.inpaint(image_rgb, prep_mask, prompt=req.prompt)
    except MemoryError:
        # Fall back to LaMa on GPU OOM.
        from backend.restoration.providers.lama import LamaProvider

        logger.warning("Inpaint OOM — falling back to LaMa")
        provider = LamaProvider()
        restored_rgb = await provider.inpaint(image_rgb, prep_mask, prompt=req.prompt)
        provider_name = "lama"

    duration_ms = int((time.perf_counter() - start) * 1000)

    restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
    out_path = Path(base_path).parent / "restored.jpg"
    cv2.imwrite(str(out_path), restored_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    await results_store.update_image_path(analysis_id, "restored", str(out_path))

    base_url_hint = settings.api_prefix
    return RestoreResponse(
        analysis_id=analysis_id,
        restored_url=f"{base_url_hint}/results/{analysis_id}/image/restored",
        provider=provider_name,
        duration_ms=duration_ms,
    )
