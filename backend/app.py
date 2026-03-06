"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routers import damage, materials, pipeline, segmentation
from backend.schemas.responses import HealthResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Building Analyzer API — analyzes building facade images using a modular "
        "ML pipeline: facade segmentation, damage detection, and material classification."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
_prefix = settings.api_prefix
app.include_router(segmentation.router, prefix=_prefix)
app.include_router(damage.router, prefix=_prefix)
app.include_router(materials.router, prefix=_prefix)
app.include_router(pipeline.router, prefix=_prefix)


@app.get(
    f"{_prefix}/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Return the API status and version."""
    return HealthResponse(status="ok", version=settings.app_version)
