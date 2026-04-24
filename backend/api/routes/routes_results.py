"""GET /api/results/... — fetch saved payloads, images, masks, restored photo."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.core.results_store import results_store

router = APIRouter()


@router.get("/results/{analysis_id}")
async def get_analysis(analysis_id: str):
    rec = await results_store.get_analysis(analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return rec.payload


@router.get("/results/{analysis_id}/images")
async def list_images(analysis_id: str):
    rec = await results_store.get_analysis(analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"images": sorted(rec.image_paths.keys())}


@router.get("/results/{analysis_id}/image/{key}")
async def get_image(analysis_id: str, key: str):
    rec = await results_store.get_analysis(analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    paths = rec.image_paths or {}
    path = paths.get(key)
    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"Image key '{key}' not found. Available: {sorted(paths.keys())}",
        )
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing on disk")
    media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return FileResponse(path, media_type=media)


# Legacy endpoint (old Flutter clients)
@router.get("/results/{analysis_id}/images/{image_type}")
async def get_image_legacy(analysis_id: str, image_type: str):
    return await get_image(analysis_id, f"viz_{image_type}")
