"""Persistence layer for scraped price snapshots and run status."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import desc, func, select, update

from backend.core.db import PriceSnapshot, ScraperRun, get_session_factory
from backend.scraper.base import ScrapedItem


async def save_snapshots(items: Iterable[ScrapedItem]) -> int:
    Session = get_session_factory()
    count = 0
    async with Session() as s:
        for it in items:
            s.add(
                PriceSnapshot(
                    source=it.source,
                    category=it.category,
                    item_name=it.item_name,
                    unit=it.unit,
                    price_rub=it.price_rub,
                    meta={**it.meta, "kind": it.kind},
                )
            )
            count += 1
        await s.commit()
    return count


async def record_run_start(source: str) -> int:
    Session = get_session_factory()
    async with Session() as s:
        run = ScraperRun(source=source, status="running")
        s.add(run)
        await s.commit()
        await s.refresh(run)
        return run.id


async def record_run_finish(
    run_id: int,
    status: str,
    items_count: int = 0,
    error_msg: str | None = None,
) -> None:
    Session = get_session_factory()
    async with Session() as s:
        await s.execute(
            update(ScraperRun)
            .where(ScraperRun.id == run_id)
            .values(
                status=status,
                items_count=items_count,
                error_msg=error_msg,
                finished_at=datetime.now(timezone.utc),
            )
        )
        await s.commit()


async def recent_failures(source: str, window_hours: int) -> int:
    """Count consecutive errors in the last `window_hours`."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    Session = get_session_factory()
    async with Session() as s:
        rows = (
            (
                await s.execute(
                    select(ScraperRun)
                    .where(
                        ScraperRun.source == source,
                        ScraperRun.started_at >= cutoff,
                    )
                    .order_by(desc(ScraperRun.started_at))
                )
            )
            .scalars()
            .all()
        )
    count = 0
    for r in rows:
        if r.status == "error" or r.status == "blocked":
            count += 1
        else:
            break
    return count


async def latest_snapshot_time() -> datetime | None:
    Session = get_session_factory()
    async with Session() as s:
        row = (
            await s.execute(select(func.max(PriceSnapshot.scraped_at)))
        ).scalar()
    return row
