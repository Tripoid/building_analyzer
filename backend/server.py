"""
FastAPI server for building facade analysis.
Runs locally or in Google Colab with ngrok tunnel.
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ml_pipeline import FacadeAnalyzer
from repair_calculator import RepairCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer: Optional[FacadeAnalyzer] = None
calculator = RepairCalculator(total_area_m2=450.0)
results_store: dict = {}  # In-memory storage for analysis results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup."""
    global analyzer
    logger.info("Starting facade analysis server...")
    analyzer = FacadeAnalyzer()
    analyzer.load_models()
    logger.info("Server ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Facade Analyzer API",
    description="ML-powered building facade analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Facade Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /api/health",
            "analyze": "POST /api/analyze (multipart/form-data, field: file)",
            "images": "GET /api/results/{id}/images/{type}",
        }
    }


@app.get("/api/health")
async def health_check():
    """Check server status and model readiness."""
    return {
        "status": "ok",
        "models_loaded": analyzer is not None and analyzer.models_loaded,
        "device": analyzer.device if analyzer else "unknown",
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    total_area_m2: float = 450.0,
):
    """
    Upload a facade photo for analysis.
    Returns full analysis results with defects, materials, and cost estimation.
    """
    if analyzer is None or not analyzer.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(image_bytes) > 50_000_000:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        logger.info(f"Analyzing image: {file.filename} ({len(image_bytes)} bytes)")

        # Run ML pipeline
        result = analyzer.analyze(image_bytes)
        analysis_id = result["id"]

        # Calculate repair costs
        calc = RepairCalculator(total_area_m2=total_area_m2)
        repair_estimate = calc.calculate(
            damages=result["damages"],
            total_area_px=result["total_area_px"],
            layer_analysis=result.get("layer_analysis", {}),
        )

        # Store for later image retrieval
        results_store[analysis_id] = {
            "image_paths": result["image_paths"],
        }

        # Build response
        response = {
            "id": analysis_id,
            "overall_score": result["overall_score"],
            "overall_condition": result["overall_condition"],
            "total_area_m2": total_area_m2,
            "total_area_px": result["total_area_px"],
            "damaged_area_px": result["damaged_area_px"],
            "damaged_area_m2": round(
                (result["damaged_area_px"] / result["total_area_px"] * total_area_m2)
                if result["total_area_px"] > 0 else 0, 1
            ),
            "damages": result["damages"],
            "materials": result["materials"],
            "layer_analysis": {
                k: {
                    "area_px": v["area_px"],
                    "affected_layers": v["affected_layers"],
                    "crack_depth": v.get("crack_depth"),
                }
                for k, v in result.get("layer_analysis", {}).items()
            },
            "repair_estimate": repair_estimate,
            "processed_images": result["processed_images"],
        }

        logger.info(f"Analysis complete: score={result['overall_score']}, id={analysis_id}")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/results/{analysis_id}/images/{image_type}")
async def get_result_image(analysis_id: str, image_type: str):
    """Get a processed image from analysis results."""
    if analysis_id not in results_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    paths = results_store[analysis_id].get("image_paths", {})
    if image_type not in paths:
        raise HTTPException(
            status_code=404,
            detail=f"Image type '{image_type}' not found. Available: {list(paths.keys())}"
        )

    path = paths[image_type]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/results/{analysis_id}/images")
async def list_result_images(analysis_id: str):
    """List available processed images for an analysis."""
    if analysis_id not in results_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    paths = results_store[analysis_id].get("image_paths", {})
    return {"images": list(paths.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
