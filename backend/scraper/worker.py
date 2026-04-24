"""
Scraper scheduler + CLI.

Runs the configured sources on a cron schedule inside the uvicorn process
(APScheduler), *and* exposes a CLI entrypoint for one-off runs or systemd/cron.

Examples
--------

    # run once inside a notebook shell
    python -m backend.scraper.worker run-once --source petrovich

    # run all enabled sources + exit (non-zero on failure)
    python -m backend.scraper.worker run-once --source all

    # print the freshness status of the price store
    python -m backend.scraper.worker status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from typing import Iterable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.core.config import get_settings
from backend.core.db import init_db
from backend.core.logging import configure_logging
from backend.scraper.base import BaseSource
from backend.scraper.sources.petrovich import PetrovichSource
from backend.scraper.sources.profi import ProfiSource
from backend.scraper.store import (
    latest_snapshot_time,
    recent_failures,
    record_run_finish,
    record_run_start,
    save_snapshots,
)

logger = logging.getLogger(__name__)

SOURCES: dict[str, type[BaseSource]] = {
    "petrovich": PetrovichSource,
    "profi": ProfiSource,
}


async def _run_source(name: str) -> tuple[str, int, str | None]:
    settings = get_settings()
    fails = await recent_failures(name, settings.scraper_circuit_breaker_hours)
    if fails >= settings.scraper_circuit_breaker_threshold:
        logger.warning(
            "circuit-breaker: skipping %s (%d recent failures within %dh)",
            name, fails, settings.scraper_circuit_breaker_hours,
        )
        return "blocked", 0, f"circuit_breaker ({fails} failures)"

    run_id = await record_run_start(name)
    try:
        Source = SOURCES[name]
        async with Source() as src:
            items = await src.fetch_all()
        count = await save_snapshots(items)
        await record_run_finish(run_id, "ok", count)
        logger.info("%s: saved %d price rows", name, count)
        return "ok", count, None
    except Exception as e:
        logger.exception("%s: scrape failed", name)
        await record_run_finish(run_id, "error", 0, str(e))
        return "error", 0, str(e)


async def run_once(source: str = "all") -> int:
    await init_db()
    settings = get_settings()
    names: Iterable[str]
    names = SOURCES.keys() if source == "all" else [source]
    any_ok = False
    for n in names:
        if n not in SOURCES:
            logger.error("unknown source: %s", n)
            continue
        status, _, _ = await _run_source(n)
        if status == "ok":
            any_ok = True
    return 0 if any_ok else 1


async def status() -> None:
    await init_db()
    ts = await latest_snapshot_time()
    if ts is None:
        print("Price store is empty — YAML baseline will be used.")
        return
    age = datetime.now(timezone.utc) - ts
    print(f"Newest price snapshot: {ts.isoformat()}  (age: {age})")


# ─────────────── APScheduler (used from create_app lifespan) ───────────────


def _scheduled_job_factory(source: str):
    async def _job():
        await _run_source(source)

    return _job


def start_scheduler() -> AsyncIOScheduler:
    configure_logging()
    settings = get_settings()
    sched = AsyncIOScheduler(timezone="UTC")
    for name in settings.scraper_sources:
        if name not in SOURCES:
            continue
        trigger = CronTrigger.from_crontab(settings.scraper_cron)
        sched.add_job(_scheduled_job_factory(name), trigger, id=f"scrape_{name}")
    sched.start()
    logger.info(
        "Scraper scheduler started for %s (%s)",
        settings.scraper_sources, settings.scraper_cron,
    )
    return sched


# ─────────────── CLI ───────────────


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="backend.scraper.worker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run-once", help="run one or all sources and exit")
    r.add_argument("--source", default="all", help="petrovich | profi | all")

    sub.add_parser("status", help="print freshness of the price store")

    args = parser.parse_args()
    if args.cmd == "run-once":
        return asyncio.run(run_once(args.source))
    if args.cmd == "status":
        asyncio.run(status())
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
