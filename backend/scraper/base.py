"""
Common scaffolding for price sources.

Each concrete `BaseSource` exposes a single `async fetch_all() -> list[ScrapedItem]`
method that the worker then writes into SQLite. The base handles:

    - User-Agent rotation
    - per-domain semaphore (default concurrency = 3)
    - polite jitter between requests
    - tenacity retries with exp backoff
    - circuit-breaker state consulted by the worker (not enforced here)

We use `curl_cffi` for HTTP to imitate a real Chrome TLS/JA3 fingerprint —
petrovich.ru sits behind a mild anti-bot that denies plain httpx.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from curl_cffi.requests import AsyncSession
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

USER_AGENTS = [
    # Modern Chrome fingerprints
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
]


@dataclass
class ScrapedItem:
    source: str
    category: str            # matches price-book key, e.g. "facade_paint"
    item_name: str
    unit: str                # m2 / kg / L / m / piece
    price_rub: float
    kind: str = "material"   # "material" | "labor"
    meta: dict[str, Any] = field(default_factory=dict)


class ScraperError(RuntimeError):
    pass


class BaseSource(ABC):
    name: str = "base"
    base_url: str = ""
    concurrency: int = 3
    min_delay: float = 0.5
    max_delay: float = 2.0

    def __init__(self):
        self._sem = asyncio.Semaphore(self.concurrency)
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "BaseSource":
        self._session = AsyncSession(
            impersonate="chrome124",
            timeout=30,
            headers={
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,*/*;q=0.8"
                ),
            },
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def fetch(self, url: str, **kwargs) -> str:
        assert self._session is not None, "call inside `async with` block"
        async with self._sem:
            await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1.5, min=1, max=15),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    resp = await self._session.get(url, **kwargs)
                    if resp.status_code >= 500:
                        raise ScraperError(f"{self.name} {url}: HTTP {resp.status_code}")
                    if resp.status_code == 403 or resp.status_code == 429:
                        raise ScraperError(
                            f"{self.name} blocked: HTTP {resp.status_code}"
                        )
                    resp.raise_for_status()
                    return resp.text
        raise ScraperError("unreachable")

    @abstractmethod
    async def fetch_all(self) -> list[ScrapedItem]:
        ...
