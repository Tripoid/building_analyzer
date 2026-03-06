"""Pipeline orchestrator module.

Combines facade segmentation, damage detection, and material classification into
a single end-to-end analysis call.
"""

from ml.pipeline.pipeline import BuildingAnalysisPipeline, PipelineConfig
from ml.pipeline.result import BuildingAnalysisResult

__all__ = [
    "BuildingAnalysisPipeline",
    "PipelineConfig",
    "BuildingAnalysisResult",
]
