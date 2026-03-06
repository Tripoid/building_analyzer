"""Backend application configuration.

Settings are loaded from environment variables (or a ``.env`` file) and can be
overridden in tests by setting the corresponding env-vars before import.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, resolved from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ---- API ---------------------------------------------------------------
    app_title: str = "Building Analyzer API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # ---- Device ------------------------------------------------------------
    device: str = "auto"

    # ---- Segmentation model ------------------------------------------------
    seg_model_name: str = "unet"
    seg_model_weights: str | None = None
    seg_image_size_h: int = 512
    seg_image_size_w: int = 512
    seg_tta: bool = False

    # ---- Damage detection model --------------------------------------------
    det_model_name: str = "anchor_free"
    det_model_weights: str | None = None
    det_image_size_h: int = 640
    det_image_size_w: int = 640
    det_score_threshold: float = 0.4
    det_nms_iou_threshold: float = 0.5
    det_max_detections: int = 100

    # ---- Material classification model -------------------------------------
    mat_model_name: str = "cnn_classifier"
    mat_model_weights: str | None = None
    mat_image_size_h: int = 224
    mat_image_size_w: int = 224
    mat_batch_size: int = 16

    @property
    def seg_image_size(self) -> tuple[int, int]:
        return (self.seg_image_size_h, self.seg_image_size_w)

    @property
    def det_image_size(self) -> tuple[int, int]:
        return (self.det_image_size_h, self.det_image_size_w)

    @property
    def mat_image_size(self) -> tuple[int, int]:
        return (self.mat_image_size_h, self.mat_image_size_w)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
