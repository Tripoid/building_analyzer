"""
Persistent store for analysis results.

Why a store instead of an in-memory dict: the notebook kernel can be restarted
(or the notebook switched), and user-uploaded analyses should survive that.
Flutter also polls this store for the async loading screen.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from backend.core.db import AnalysisRecord, CalibrationRecord, get_session_factory


class ResultsStore:
    # Analyses --------------------------------------------------------------

    async def put_analysis(
        self, analysis_id: str, payload: dict[str, Any], image_paths: dict[str, str]
    ) -> None:
        Session = get_session_factory()
        async with Session() as s:
            existing = await s.get(AnalysisRecord, analysis_id)
            if existing is None:
                s.add(
                    AnalysisRecord(
                        id=analysis_id, payload=payload, image_paths=image_paths
                    )
                )
            else:
                existing.payload = payload
                existing.image_paths = image_paths
            await s.commit()

    async def get_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        Session = get_session_factory()
        async with Session() as s:
            return await s.get(AnalysisRecord, analysis_id)

    async def delete_analysis(self, analysis_id: str) -> None:
        Session = get_session_factory()
        async with Session() as s:
            await s.execute(
                delete(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
            )
            await s.commit()

    async def update_image_path(
        self, analysis_id: str, key: str, path: str
    ) -> None:
        Session = get_session_factory()
        async with Session() as s:
            rec = await s.get(AnalysisRecord, analysis_id)
            if rec is None:
                return
            paths = dict(rec.image_paths or {})
            paths[key] = path
            rec.image_paths = paths
            await s.commit()

    # Calibrations ----------------------------------------------------------

    async def put_calibration(self, calibration_id: str, payload: dict[str, Any]) -> None:
        Session = get_session_factory()
        async with Session() as s:
            existing = await s.get(CalibrationRecord, calibration_id)
            if existing is None:
                s.add(CalibrationRecord(id=calibration_id, payload=payload))
            else:
                existing.payload = payload
            await s.commit()

    async def get_calibration(self, calibration_id: str) -> dict[str, Any] | None:
        Session = get_session_factory()
        async with Session() as s:
            rec = await s.get(CalibrationRecord, calibration_id)
            return rec.payload if rec else None


results_store = ResultsStore()
