"""
Multi-layer "repair pie" logic.

A defect is not a single line item — it unfolds into a *stack* of sequential
repair steps (demolition → substrate → primer → plaster → putty → paint → ...).
Each step consumes N materials and one labour activity. Different defects trigger
different stacks; here they live in one declarative table so the calculator stays
trivial.

Units:
    - m2 (area) for surface work
    - m   (linear) for crack sealing
    - unit (шт) for discrete items such as window replacement

All prices are looked up in RUB from the `scraper` package — this file only knows
about *consumption rates* and *step names*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Unit = Literal["m2", "m", "unit", "kg", "L", "piece"]


@dataclass(frozen=True)
class MaterialConsumption:
    material_key: str           # matches price catalog / YAML
    display_ru: str
    unit: Unit                  # unit the price is stored in
    rate: float                 # consumption per *defect* unit (m² / m / piece)


@dataclass(frozen=True)
class StepSpec:
    key: str
    display_ru: str
    # Work is charged per the *defect* unit (crack is m, surface is m²).
    labor_key: str
    labor_display_ru: str
    unit: Unit                  # the defect unit
    norm_hours: float           # per defect unit
    materials: tuple[MaterialConsumption, ...]


# ─────────────────────── Step catalog ───────────────────────

STEP_CATALOG: dict[str, StepSpec] = {
    "brick_restoration": StepSpec(
        key="brick_restoration",
        display_ru="Восстановление кирпичной кладки",
        labor_key="masonry",
        labor_display_ru="Кладочные работы",
        unit="m2",
        norm_hours=2.5,
        materials=(
            MaterialConsumption("cement_mix", "Цементно-песчаная смесь", "kg", 12.0),
            MaterialConsumption("facade_brick", "Кирпич фасадный", "piece", 50.0),
        ),
    ),
    "crack_repair_mortar": StepSpec(
        key="crack_repair_mortar",
        display_ru="Заделка трещин ремонтным составом",
        labor_key="crack_repair",
        labor_display_ru="Ремонт трещин",
        unit="m",
        norm_hours=0.8,
        materials=(
            MaterialConsumption("crack_sealant", "Ремонтный состав (цементный)", "kg", 1.5),
            MaterialConsumption("fiber_mesh", "Фиброармирующая лента", "m", 1.1),
        ),
    ),
    "primer": StepSpec(
        key="primer",
        display_ru="Грунтование",
        labor_key="priming",
        labor_display_ru="Грунтование поверхности",
        unit="m2",
        norm_hours=0.2,
        materials=(
            MaterialConsumption("facade_primer", "Грунтовка фасадная глубокого проникновения", "L", 0.20),
        ),
    ),
    "plaster": StepSpec(
        key="plaster",
        display_ru="Штукатурные работы",
        labor_key="plastering",
        labor_display_ru="Оштукатуривание",
        unit="m2",
        norm_hours=2.0,
        materials=(
            MaterialConsumption("facade_plaster", "Штукатурка фасадная", "kg", 16.0),
            MaterialConsumption("fiber_mesh_sq", "Сетка армирующая", "m2", 1.1),
        ),
    ),
    "putty": StepSpec(
        key="putty",
        display_ru="Шпаклевание",
        labor_key="puttying",
        labor_display_ru="Шпаклевание поверхности",
        unit="m2",
        norm_hours=0.5,
        materials=(
            MaterialConsumption("facade_putty", "Шпатлёвка фасадная", "kg", 0.8),
        ),
    ),
    "paint": StepSpec(
        key="paint",
        display_ru="Покраска фасадной краской",
        labor_key="painting",
        labor_display_ru="Покраска фасада",
        unit="m2",
        norm_hours=0.4,
        materials=(
            MaterialConsumption("facade_paint", "Краска фасадная", "L", 0.25),
        ),
    ),
    "biocide_wash": StepSpec(
        key="biocide_wash",
        display_ru="Биоцидная обработка от высолов/моха",
        labor_key="biocide_treatment",
        labor_display_ru="Обработка биоцидом",
        unit="m2",
        norm_hours=0.3,
        materials=(
            MaterialConsumption("biocide", "Биоцидный состав / антисоль", "L", 0.30),
        ),
    ),
    "rust_converter": StepSpec(
        key="rust_converter",
        display_ru="Антикоррозийная обработка",
        labor_key="rust_treatment",
        labor_display_ru="Обработка преобразователем ржавчины",
        unit="m2",
        norm_hours=0.4,
        materials=(
            MaterialConsumption("rust_converter_mat", "Преобразователь ржавчины", "L", 0.15),
            MaterialConsumption("anticorrosion_primer", "Грунт антикоррозийный", "L", 0.12),
        ),
    ),
    "metal_paint": StepSpec(
        key="metal_paint",
        display_ru="Покраска по металлу",
        labor_key="metal_painting",
        labor_display_ru="Окраска металлических элементов",
        unit="m2",
        norm_hours=0.6,
        materials=(
            MaterialConsumption("metal_paint_mat", "Краска по металлу", "L", 0.14),
        ),
    ),
    "antifungal": StepSpec(
        key="antifungal",
        display_ru="Антигрибковая обработка",
        labor_key="antifungal_treatment",
        labor_display_ru="Обработка от плесени",
        unit="m2",
        norm_hours=0.3,
        materials=(
            MaterialConsumption("antifungal_mat", "Антигрибковый состав", "L", 0.25),
        ),
    ),
    "wood_removal": StepSpec(
        key="wood_removal",
        display_ru="Демонтаж поражённой древесины",
        labor_key="wood_demolition",
        labor_display_ru="Демонтаж деревянных элементов",
        unit="m2",
        norm_hours=0.8,
        materials=(),
    ),
    "wood_impregnation": StepSpec(
        key="wood_impregnation",
        display_ru="Пропитка антисептиком",
        labor_key="wood_impregnation",
        labor_display_ru="Антисептическая пропитка",
        unit="m2",
        norm_hours=0.3,
        materials=(
            MaterialConsumption("wood_antiseptic", "Антисептик для дерева", "L", 0.25),
        ),
    ),
    "wood_replacement": StepSpec(
        key="wood_replacement",
        display_ru="Замена деревянных элементов",
        labor_key="wood_installation",
        labor_display_ru="Монтаж деревянных элементов",
        unit="m2",
        norm_hours=1.5,
        materials=(
            MaterialConsumption("timber_board", "Доска обрезная / террасная", "m2", 1.05),
        ),
    ),
    "wood_paint": StepSpec(
        key="wood_paint",
        display_ru="Покраска / лакировка по дереву",
        labor_key="wood_painting",
        labor_display_ru="Покраска деревянных элементов",
        unit="m2",
        norm_hours=0.5,
        materials=(
            MaterialConsumption("wood_paint_mat", "Краска / лак по дереву", "L", 0.18),
        ),
    ),
    "glass_replacement": StepSpec(
        key="glass_replacement",
        display_ru="Замена остекления",
        labor_key="glazing",
        labor_display_ru="Остекление",
        unit="m2",
        norm_hours=1.2,
        materials=(
            MaterialConsumption("window_glass", "Стеклопакет", "m2", 1.02),
            MaterialConsumption("window_sealant", "Герметик оконный", "L", 0.05),
        ),
    ),
}


# ─────────────────────── Defect → recipe ───────────────────────

DEFECT_RECIPES: dict[str, list[str]] = {
    "exposed_brick":   ["brick_restoration", "primer", "plaster", "putty", "paint"],
    "crack_deep":      ["crack_repair_mortar", "primer", "plaster", "putty", "paint"],
    "crack_surface":   ["crack_repair_mortar", "putty", "paint"],
    "crack":           ["crack_repair_mortar", "putty", "paint"],  # default
    "peeling":         ["putty", "paint"],
    "spalling":        ["plaster", "putty", "paint"],
    "efflorescence":   ["biocide_wash", "primer", "paint"],
    "rust":            ["rust_converter", "primer", "metal_paint"],
    "rust_stain":      ["rust_converter", "primer", "metal_paint"],
    "water_damage":    ["antifungal", "primer", "putty", "paint"],
    "moss":            ["biocide_wash", "primer", "paint"],
    "mold":            ["antifungal", "primer", "paint"],
    "wood_rot":        ["wood_removal", "wood_impregnation", "wood_replacement", "wood_paint"],
    "damaged_wood":    ["wood_impregnation", "wood_paint"],
    "broken_glass":    ["glass_replacement"],
    "glass_crack":     ["glass_replacement"],
    "rusty_metal":     ["rust_converter", "metal_paint"],
    "damaged_railing": ["rust_converter", "metal_paint"],
}


def recipe_for(defect_type: str, crack_depth: str | None = None) -> list[StepSpec]:
    """Resolve a defect (with optional crack depth) into ordered step specs."""
    if defect_type == "crack" and crack_depth:
        key = f"crack_{crack_depth}"
    else:
        key = defect_type
    step_keys = DEFECT_RECIPES.get(key, DEFECT_RECIPES.get(defect_type, []))
    return [STEP_CATALOG[k] for k in step_keys if k in STEP_CATALOG]
