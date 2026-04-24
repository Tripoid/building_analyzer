"""
profi.ru — labour price aggregator.

profi.ru exposes per-category "specialist listings" with an average tariff.
We parse the "от X ₽/м²" hint off the category landing page — each key below
maps to a concrete category URL.

If profi.ru starts to block us (it does rotate anti-bot checks), save_snapshots
will simply record an error and the YAML baseline takes over.
"""

from __future__ import annotations

import logging
import re
from statistics import median

from selectolax.lexbor import LexborHTMLParser

from backend.scraper.base import BaseSource, ScrapedItem, ScraperError

logger = logging.getLogger(__name__)

# Labour price-book key → (category URL path, unit).
LABOR_CATEGORIES: dict[str, tuple[str, str]] = {
    "painting":              ("/remont/pokraska-sten/",                 "m2"),
    "plastering":            ("/remont/shtukaturka-sten/",              "m2"),
    "priming":               ("/remont/gruntovka-sten/",                "m2"),
    "puttying":              ("/remont/shpaklyovka-sten/",              "m2"),
    "masonry":               ("/remont/kladka-kirpicha/",               "m2"),
    "crack_repair":          ("/remont/zadelka-treschin/",              "m"),
    "biocide_treatment":     ("/remont/antiseptirovanie-sten/",         "m2"),
    "rust_treatment":        ("/remont/antikorroziynaya-obrabotka/",    "m2"),
    "metal_painting":        ("/remont/pokraska-metalla/",              "m2"),
    "antifungal_treatment":  ("/remont/udalenie-pleseni/",              "m2"),
    "wood_demolition":       ("/remont/demontazh-dereva/",              "m2"),
    "wood_impregnation":     ("/remont/propitka-dereva/",               "m2"),
    "wood_installation":     ("/remont/montazh-doski/",                 "m2"),
    "wood_painting":         ("/remont/pokraska-dereva/",               "m2"),
    "glazing":               ("/okna/zamena-stekol/",                   "m2"),
    "scaffolding_per_floor": ("/stroitelstvo/montazh-lesov/",           "floor"),
}

_PRICE_RE = re.compile(r"(?:от|from)?\s?(\d[\d\s]+)\s?(?:руб|₽)\s?/\s?(?:м²|м2|кв\.?м|пог\.?м|эт)", re.I)


def _extract_prices(html: str) -> list[float]:
    parser = LexborHTMLParser(html)
    prices: list[float] = []

    # Profi often renders a "от 500 ₽ / м²" badge on the category hero.
    for node in parser.css(".price, .price__value, [data-qa='price']"):
        text = node.text(strip=True)
        for m in _PRICE_RE.finditer(text):
            prices.append(float(m.group(1).replace(" ", "")))

    # Fallback: scan all text on the page
    if not prices:
        text = parser.css_first("body").text(separator=" ") if parser.css_first("body") else ""
        for m in _PRICE_RE.finditer(text):
            prices.append(float(m.group(1).replace(" ", "")))
    return prices


class ProfiSource(BaseSource):
    name = "profi"
    base_url = "https://profi.ru"
    concurrency = 2

    async def fetch_all(self) -> list[ScrapedItem]:
        results: list[ScrapedItem] = []
        for key, (path, unit) in LABOR_CATEGORIES.items():
            url = f"{self.base_url}{path}"
            try:
                html = await self.fetch(url)
            except ScraperError as e:
                logger.warning("profi %s: %s", key, e)
                continue
            prices = _extract_prices(html)
            if not prices:
                logger.warning("profi %s: no prices parsed", key)
                continue
            med = round(median(prices), 2)
            results.append(
                ScrapedItem(
                    source=self.name,
                    category=key,
                    item_name=f"profi median[{len(prices)}]",
                    unit=unit,
                    price_rub=med,
                    kind="labor",
                    meta={"samples": len(prices), "url": url},
                )
            )
        return results
