"""Structured result types for the building analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SegmentationSummary:
    """Summary of the facade segmentation stage."""

    class_area_fractions: dict[str, float] = field(default_factory=dict)
    dominant_class: str = "unknown"
    damaged_area_fraction: float = 0.0


@dataclass
class DamageInstance:
    """A single detected damage instance."""

    label: int
    label_name: str
    score: float
    box: list[float]  # [x1, y1, x2, y2]
    material_in_region: str | None = None
    material_score: float | None = None


@dataclass
class MaterialSummary:
    """Summary of the material classification stage."""

    overall_dominant_material: str | None = None
    intact_material: str | None = None
    damaged_material: str | None = None
    region_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildingAnalysisResult:
    """Structured output of the full building facade analysis pipeline.

    Attributes:
        image_path: Source image path (if known).
        segmentation: Summary from the facade segmentation stage.
        damage_instances: List of detected damage regions with metadata.
        materials: Summary from the material classification stage.
        num_damage_instances: Total number of detected damage instances.
        metadata: Arbitrary key-value pairs for provenance tracking.
    """

    image_path: str | None = None
    segmentation: SegmentationSummary = field(default_factory=SegmentationSummary)
    damage_instances: list[DamageInstance] = field(default_factory=list)
    materials: MaterialSummary = field(default_factory=MaterialSummary)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_damage_instances(self) -> int:
        return len(self.damage_instances)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return {
            "image_path": self.image_path,
            "segmentation": {
                "class_area_fractions": self.segmentation.class_area_fractions,
                "dominant_class": self.segmentation.dominant_class,
                "damaged_area_fraction": self.segmentation.damaged_area_fraction,
            },
            "num_damage_instances": self.num_damage_instances,
            "damage_instances": [
                {
                    "label": inst.label,
                    "label_name": inst.label_name,
                    "score": inst.score,
                    "box": inst.box,
                    "material_in_region": inst.material_in_region,
                    "material_score": inst.material_score,
                }
                for inst in self.damage_instances
            ],
            "materials": {
                "overall_dominant_material": self.materials.overall_dominant_material,
                "intact_material": self.materials.intact_material,
                "damaged_material": self.materials.damaged_material,
            },
            "metadata": self.metadata,
        }
