"""FastAPI dependency providers: shared singletons (analyzer, inpainter, store)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from backend.ml_pipeline import FacadeAnalyzer
    from backend.restoration.providers.base import InpaintProvider


_analyzer: "FacadeAnalyzer | None" = None
_inpainter: "InpaintProvider | None" = None


def set_analyzer(analyzer: "FacadeAnalyzer") -> None:
    global _analyzer
    _analyzer = analyzer


def get_analyzer() -> "FacadeAnalyzer":
    if _analyzer is None or not getattr(_analyzer, "models_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet — wait for startup to complete",
        )
    return _analyzer


def get_inpainter() -> "InpaintProvider":
    global _inpainter
    if _inpainter is None:
        from backend.core.config import get_settings
        from backend.restoration.providers.lama import LamaProvider
        from backend.restoration.providers.sd_inpaint import SDInpaintProvider

        provider_name = get_settings().inpaint_provider
        if provider_name == "sd":
            _inpainter = SDInpaintProvider()
        else:
            _inpainter = LamaProvider()
    return _inpainter


def reset_inpainter() -> None:
    """Used when switching providers at runtime or after OOM."""
    global _inpainter
    _inpainter = None
