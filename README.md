# Building Analyzer

A production-quality ML system that analyzes building facades from images.

## Overview

The system performs:
- **Facade segmentation** — detects structural elements (walls, windows, doors, balconies, cornices) and separates intact from damaged segments
- **Damage detection** — detects and highlights damaged regions with bounding boxes and masks
- **Material classification** — identifies building materials in both intact and damaged regions
- **Full pipeline** — orchestrates all modules into a single structured analysis result

## Architecture

```
building_analyzer/
├── ml/
│   ├── common/                    # Shared abstractions (base models, datasets, transforms)
│   ├── facade_segmentation/       # Structural element segmentation module
│   ├── damage_detection/          # Damage region detection module
│   ├── material_classification/   # Material identification module
│   └── pipeline/                  # End-to-end pipeline orchestrator
├── backend/
│   ├── routers/                   # FastAPI route handlers per module
│   ├── schemas/                   # Pydantic request/response schemas
│   └── app.py                     # FastAPI application entry point
└── notebooks/                     # Google Colab notebooks for each module
```

## Key Design Principles

- **Model-agnostic interfaces** — swap any model implementation without changing downstream code
- **Modular** — each ML stage is independently trainable, testable, and deployable
- **Reproducible** — every module ships with a Colab notebook replicating its functionality
- **Scalable** — registry pattern allows adding new model architectures at runtime

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"
```

### Run the API server

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Run tests

```bash
pytest
```

### Run tests with coverage

```bash
pytest --cov=ml --cov=backend --cov-report=html
```

## ML Modules

### Facade Segmentation (`ml/facade_segmentation/`)

Semantic segmentation model that labels each pixel as one of:
`background`, `wall`, `window`, `door`, `balcony`, `cornice`, `damaged`

| File | Purpose |
|------|---------|
| `dataset.py` | `FacadeSegmentationDataset` — loads images + segmentation masks |
| `model.py` | `SegmentationModel` — pluggable encoder-decoder architecture |
| `train.py` | `SegmentationTrainer` — full training loop with checkpointing |
| `inference.py` | `SegmentationInferencer` — single-image and batch inference |
| `utils.py` | Colour palettes, mask overlays, IoU helpers |

### Damage Detection (`ml/damage_detection/`)

Instance segmentation / object detection model that produces per-damage bounding boxes, masks, and confidence scores.

| File | Purpose |
|------|---------|
| `dataset.py` | `DamageDetectionDataset` — loads images + COCO-style annotations |
| `model.py` | `DamageDetectionModel` — pluggable detection backbone |
| `train.py` | `DamageDetectionTrainer` — training loop with mAP evaluation |
| `inference.py` | `DamageDetectionInferencer` — inference with NMS post-processing |
| `utils.py` | Bounding-box helpers, mask utilities, visualisation |

### Material Classification (`ml/material_classification/`)

Region-based classification model that identifies materials (concrete, brick, glass, wood, metal, stone) for arbitrary image crops.

| File | Purpose |
|------|---------|
| `dataset.py` | `MaterialDataset` — image-crop + label dataset |
| `model.py` | `MaterialClassifier` — pluggable classification backbone |
| `train.py` | `MaterialTrainer` — training loop with per-class accuracy |
| `inference.py` | `MaterialInferencer` — crop-level and region-level inference |
| `utils.py` | Crop extraction, GradCAM, result aggregation |

### Pipeline (`ml/pipeline/`)

Orchestrates all three modules into a single call returning a `BuildingAnalysisResult`.

## Backend API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/segmentation/predict` | Run facade segmentation |
| `POST` | `/api/v1/damage/predict` | Run damage detection |
| `POST` | `/api/v1/materials/predict` | Run material classification |
| `POST` | `/api/v1/pipeline/analyze` | Run full analysis pipeline |
| `GET`  | `/api/v1/health` | Health check |

All endpoints accept `multipart/form-data` image uploads and return JSON.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_facade_segmentation.ipynb` | Train and evaluate facade segmentation |
| `notebooks/02_damage_detection.ipynb` | Train and evaluate damage detection |
| `notebooks/03_material_classification.ipynb` | Train and evaluate material classifier |
| `notebooks/04_full_pipeline.ipynb` | End-to-end pipeline demonstration |

## Adding a New Model

1. Subclass the appropriate base model (e.g. `BaseSegmentationModel`)
2. Implement `forward()` and `predict()`
3. Register with `@ModelRegistry.register("my_model")`
4. Pass `model_name="my_model"` to the trainer or inferencer config

## License

MIT