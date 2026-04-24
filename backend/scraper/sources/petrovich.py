"""
petrovich.ru — construction-materials catalog source.

Strategy
--------
petrovich.ru exposes category pages with JSON-LD (application/ld+json) blocks
carrying product prices. We parse those whenever present and fall back to the
regular HTML product cards otherwise. Per category we take the *median* price
(resistant to outliers / promos) for the "base" product form.

Category map — each catalog key maps to (URL path, price-book key, unit).
"""

from __future__ import annotations

import json
import logging
import re
from statistics import median

from selectolax.lexbor import LexborHTMLParser

from backend.scraper.base import BaseSource, ScrapedItem, ScraperError

logger = logging.getLogger(__name__)

# Price-book key → catalog path (must be kept in sync with scraper/default_prices_rub.yaml)
# NOTE: petrovich.ru URLs change; these are stable "Каталог → раздел" slugs
# as of 2026-04. If the catalogue tree moves, fix the path here.
CATEGORIES: dict[str, tuple[str, str]] = {
    "facade_paint":       ("/catalog/fasadnaya-kraska/",                     "L"),
    "facade_primer":      ("/catalog/gruntovki-glubokogo-pronikonnoveniya/", "L"),
    "facade_plaster":     ("/catalog/fasadnaya-shtukaturka/",                "kg"),
    "facade_putty":       ("/catalog/fasadnaya-shpaklevka/",                 "kg"),
    "facade_brick":       ("/catalog/oblicovochnyy-kirpich/",                "piece"),
    "cement_mix":         ("/catalog/tsementno-peschanye-smesi/",            "kg"),
    "crack_sealant":      ("/catalog/remontnye-sostavy/",                    "kg"),
    "fiber_mesh":         ("/catalog/armiruyushchaya-setka/",                "m"),
    "fiber_mesh_sq":      ("/catalog/armiruyushchaya-setka/",                "m2"),
    "biocide":            ("/catalog/biotsidy-antisepti/",                   "L"),
    "rust_converter_mat": ("/catalog/preobrazovateli-rzhavchiny/",           "L"),
    "anticorrosion_primer": ("/catalog/antikorroziynyy-grunt/",              "L"),
    "metal_paint_mat":    ("/catalog/kraska-po-metallu/",                    "L"),
    "wood_antiseptic":    ("/catalog/antiseptiki-dlya-dereva/",              "L"),
    "wood_paint_mat":     ("/catalog/kraski-po-derevu/",                     "L"),
    "timber_board":       ("/catalog/obreznaya-doska/",                      "m2"),
    "antifungal_mat":     ("/catalog/sostav-ot-pleseni/",                    "L"),
    "window_glass":       ("/catalog/steklopakety/",                         "m2"),
    "window_sealant":     ("/catalog/germetik-okonnyy/",                     "L"),
}


_PRICE_RE = re.compile(r"(\d[\d\s]{1,})\s?(?:руб|₽)", re.I)


def _extract_prices(html: str) -> list[float]:
    parser = LexborHTMLParser(html)

    prices: list[float] = []

    # 1) Prefer JSON-LD blocks (clean, canonical).
    for node in parser.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.text())
        except Exception:
            continue
        for product in _iter_products(data):
            offers = product.get("offers")
            if isinstance(offers, list):
                for off in offers:
                    p = _as_float(off.get("price"))
                    if p:
                        prices.append(p)
            elif isinstance(offers, dict):
                p = _as_float(offers.get("price"))
                if p:
                    prices.append(p)

    # 2) Fallback to rendered prices on product cards
    if not prices:
        for node in parser.css('[data-test="product-price"]'):
            for m in _PRICE_RE.finditer(node.text() or ""):
                prices.append(_as_float(m.group(1).replace(" ", "")) or 0)
        prices = [p for p in prices if p > 0]

    return prices


def _iter_products(node):
    if isinstance(node, dict):
        if node.get("@type") in {"Product", "schema:Product"}:
            yield node
        for v in node.values():
            yield from _iter_products(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_products(v)


def _as_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


class PetrovichSource(BaseSource):
    name = "petrovich"
    base_url = "https://petrovich.ru"
    concurrency = 2

    async def fetch_all(self) -> list[ScrapedItem]:
        results: list[ScrapedItem] = []
        for key, (path, unit) in CATEGORIES.items():
            url = f"{self.base_url}{path}"
            try:
                html = await self.fetch(url)
            except ScraperError as e:
                logger.warning("petrovich %s: %s", key, e)
                continue
            prices = _extract_prices(html)
            if not prices:
                logger.warning("petrovich %s: no prices found", key)
                continue
            med = round(median(prices), 2)
            results.append(
                ScrapedItem(
                    source=self.name,
                    category=key,
                    item_name=f"petrovich median[{len(prices)}]",
                    unit=unit,
                    price_rub=med,
                    kind="material",
                    meta={"samples": len(prices), "url": url},
                )
            )
        return results
