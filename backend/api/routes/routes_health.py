"""GET /api/health — liveness + readiness for Flutter settings screen."""

from fastapi import APIRouter

from backend.api import deps
from backend.api.schemas import HealthResponse
from backend.core.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    s = get_settings()
    analyzer = None
    try:
        analyzer = deps.get_analyzer()
    except Exception:
        pass
    return HealthResponse(
        status="ok" if analyzer else "degraded",
        models_loaded=bool(analyzer and getattr(analyzer, "models_loaded", False)),
        device=getattr(analyzer, "device", "unknown") if analyzer else "unknown",
        version=s.version,
        inpaint_provider=s.inpaint_provider,
        scraper_enabled=s.scraper_enabled,
    )
