"""
Price lookup — reads SQLite snapshots from the scraper, falls back to the
committed YAML baseline when no fresh data is available.

Return value is a `PriceSnapshot` dict-like object with:
    - unit prices in RUB
    - source label ("live" or "yaml_fallback")
    - scraped_at  (datetime or None)
    - stale       (bool — true if data older than settings.stale_price_days)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from backend.core.config import get_settings
from backend.core.db import PriceSnapshot, get_session_factory

logger = logging.getLogger(__name__)

YAML_PATH = (
    Path(__file__).resolve().parent.parent / "scraper" / "default_prices_rub.yaml"
)


@dataclass
class PriceBook:
    materials: dict[str, dict[str, Any]]
    labor: dict[str, dict[str, Any]]
    source: str                     # "live" | "yaml_fallback" | "mixed"
    snapshot_date: datetime | None
    stale: bool

    def material_price(self, key: str) -> tuple[float, str] | None:
        item = self.materials.get(key)
        if not item:
            return None
        return float(item["price"]), str(item.get("unit", "piece"))

    def labor_price(self, key: str) -> tuple[float, str] | None:
        item = self.labor.get(key)
        if not item:
            return None
        return float(item["price"]), str(item.get("unit", "m2"))


def _load_yaml_baseline() -> tuple[dict, dict]:
    if not YAML_PATH.exists():
        logger.warning("default_prices_rub.yaml missing — all prices will be 0")
        return {}, {}
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
    return raw.get("materials", {}) or {}, raw.get("labor", {}) or {}


async def load_price_book() -> PriceBook:
    settings = get_settings()
    yaml_materials, yaml_labor = _load_yaml_baseline()

    Session = get_session_factory()
    fresh_material: dict[str, dict[str, Any]] = {}
    fresh_labor: dict[str, dict[str, Any]] = {}
    newest: datetime | None = None

    try:
        async with Session() as s:
            rows = (
                (
                    await s.execute(
                        select(PriceSnapshot).order_by(
                            PriceSnapshot.scraped_at.desc()
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                bucket = (
                    fresh_labor if (row.meta or {}).get("kind") == "labor" else fresh_material
                )
                # Only keep the *latest* snapshot per category.
                if row.category in bucket:
                    continue
                bucket[row.category] = {
                    "price": float(row.price_rub),
                    "unit": row.unit,
                    "item_name": row.item_name,
                    "source": row.source,
                    "scraped_at": row.scraped_at,
                }
                if newest is None or row.scraped_at > newest:
                    newest = row.scraped_at
    except Exception as e:
        logger.warning("SQLite unavailable (%s); falling back to YAML baseline", e)

    # Merge: live overrides YAML per-key
    materials = {**yaml_materials, **fresh_material}
    labor = {**yaml_labor, **fresh_labor}

    stale = False
    source: str
    if not fresh_material and not fresh_labor:
        source = "yaml_fallback"
        stale = True
    elif newest is None:
        source = "yaml_fallback"
        stale = True
    else:
        age = datetime.now(timezone.utc) - newest
        stale = age > timedelta(days=settings.stale_price_days)
        source = "live" if not stale else "yaml_fallback"

    return PriceBook(
        materials=materials,
        labor=labor,
        source=source,
        snapshot_date=newest,
        stale=stale,
    )
