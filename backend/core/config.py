"""
Typed settings loaded from environment variables (.env supported).
Accessed via `get_settings()` (lru_cached singleton) or via FastAPI Depends.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="ALEGRO_",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    api_prefix: str = "/api"
    version: str = "2.0.0"

    # Storage
    db_path: Path = PROJECT_ROOT / "data" / "alegrocode.db"
    upload_dir: Path = Path("/tmp/alegrocode/uploads")
    results_dir: Path = Path("/tmp/alegrocode/results")
    upload_ttl_hours: int = 24

    # ML
    max_image_bytes: int = 50_000_000
    max_image_side_px: int = 1024

    # Calibration priors (metres)
    prior_door_width: tuple[float, float] = (0.7, 1.2)
    prior_window_width: tuple[float, float] = (0.8, 2.0)
    prior_brick_width: tuple[float, float] = (0.2, 0.3)

    # Estimator
    vat_rate: float = 0.20  # РФ
    waste_factor: float = 1.10
    stale_price_days: int = 14

    # Scraper
    scraper_enabled: bool = True
    scraper_cron: str = "0 3 * * *"
    scraper_sources: list[str] = Field(default_factory=lambda: ["petrovich", "profi"])
    scraper_circuit_breaker_hours: int = 24
    scraper_circuit_breaker_threshold: int = 3

    # Inpainting
    inpaint_provider: Literal["lama", "sd"] = "lama"
    lama_model_url: str = (
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/"
        "download/v0.1.0/big-lama.pt"
    )
    sd_model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    min_free_vram_mb_for_sd: int = 4096

    # Tunnel
    ngrok_authtoken: str | None = None

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
