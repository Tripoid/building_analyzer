"""
FastAPI app factory — used by both the Jupyter notebook and the CLI entrypoint.

Usage:
    from backend.api import create_app
    app = create_app()
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import deps
from backend.api.routes import (
    routes_analyze,
    routes_calibrate,
    routes_estimate,
    routes_health,
    routes_restore,
    routes_results,
)
from backend.core.config import get_settings
from backend.core.db import init_db
from backend.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()
    await init_db()

    # Analyzer is injected lazily; the notebook preloads it into globals.
    analyzer = getattr(app.state, "analyzer", None)
    if analyzer is None:
        from backend.ml_pipeline import FacadeAnalyzer

        analyzer = FacadeAnalyzer()
        analyzer.load_models()
        app.state.analyzer = analyzer

    deps.set_analyzer(analyzer)

    # Optionally start the scraper scheduler
    if settings.scraper_enabled:
        try:
            from backend.scraper.worker import start_scheduler

            app.state.scraper_scheduler = start_scheduler()
        except Exception as e:  # never block server startup on scraper issues
            import logging

            logging.getLogger(__name__).warning("Scraper scheduler disabled: %s", e)

    yield

    sched = getattr(app.state, "scraper_scheduler", None)
    if sched is not None:
        sched.shutdown(wait=False)


def create_app(analyzer=None) -> FastAPI:
    """
    Build the FastAPI app.

    If `analyzer` is passed (e.g. from the Jupyter notebook where models
    live in the kernel globals), it is reused — no re-download of weights.
    """
    settings = get_settings()
    app = FastAPI(
        title="AlegroCode API",
        description="Facade defect detection, repair estimate and AI restoration",
        version=settings.version,
        lifespan=lifespan,
    )

    if analyzer is not None:
        app.state.analyzer = analyzer
        deps.set_analyzer(analyzer)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix
    app.include_router(routes_health.router, prefix=prefix, tags=["health"])
    app.include_router(routes_calibrate.router, prefix=prefix, tags=["calibration"])
    app.include_router(routes_analyze.router, prefix=prefix, tags=["analysis"])
    app.include_router(routes_results.router, prefix=prefix, tags=["results"])
    app.include_router(routes_estimate.router, prefix=prefix, tags=["estimate"])
    app.include_router(routes_restore.router, prefix=prefix, tags=["restoration"])

    @app.get("/")
    async def root():
        return {
            "name": "AlegroCode API",
            "version": settings.version,
            "docs": "/docs",
        }

    return app


__all__ = ["create_app"]
