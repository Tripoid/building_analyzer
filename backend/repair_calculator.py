"""
RepairCalculator — Calculates repair materials, labor, and cost estimates
based on detected defects and multi-layer wall structure.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict
import math


# ═══════════════════════════════════════════════════════
# MATERIAL DATABASE (prices in KZT — Kazakhstani Tenge)
# ═══════════════════════════════════════════════════════

@dataclass
class RepairMaterial:
    name: str
    name_display: str
    unit: str  # кг, л, м², шт, м.п.
    price_per_unit: float  # KZT
    quantity: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.price_per_unit * self.quantity


@dataclass
class RepairWork:
    name: str
    name_display: str
    unit: str  # м², м.п., шт
    price_per_unit: float  # KZT per unit (labor cost)
    quantity: float = 0.0
    norm_hours_per_unit: float = 0.0  # hours per unit

    @property
    def total_cost(self) -> float:
        return self.price_per_unit * self.quantity

    @property
    def total_hours(self) -> float:
        return self.norm_hours_per_unit * self.quantity


# Consumption rates per defect type
# Format: {defect_type: [(material_name, display, unit, rate_per_m2, price_per_unit)]}
MATERIAL_RATES = {
    "crack_surface": [
        ("facade_putty", "Шпатлёвка фасадная", "кг", 0.3, 850),
        ("primer_deep", "Грунтовка глубокого проникновения", "л", 0.15, 1200),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "crack_deep": [
        ("repair_compound", "Ремонтный состав (цементный)", "кг", 1.5, 1500),
        ("reinforcing_mesh", "Армирующая сетка фасадная", "м²", 1.1, 650),
        ("primer_deep", "Грунтовка глубокого проникновения", "л", 0.15, 1200),
        ("facade_putty", "Шпатлёвка фасадная", "кг", 0.5, 850),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "peeling": [
        ("cement_plaster", "Штукатурка цементная фасадная", "кг", 16.0, 320),
        ("reinforcing_mesh", "Армирующая сетка фасадная", "м²", 1.1, 650),
        ("primer_deep", "Грунтовка глубокого проникновения", "л", 0.2, 1200),
        ("facade_putty", "Шпатлёвка фасадная", "кг", 0.8, 850),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "exposed_brick": [
        ("cement_plaster", "Штукатурка цементная фасадная", "кг", 16.0, 320),
        ("reinforcing_mesh", "Армирующая сетка фасадная", "м²", 1.1, 650),
        ("primer_deep", "Грунтовка глубокого проникновения", "л", 0.2, 1200),
        ("facade_putty", "Шпатлёвка фасадная", "кг", 0.8, 850),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "water_damage": [
        ("waterproof_compound", "Гидроизоляционный состав", "кг", 1.2, 2800),
        ("primer_deep", "Грунтовка глубокого проникновения", "л", 0.2, 1200),
        ("cement_plaster", "Штукатурка цементная фасадная", "кг", 8.0, 320),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "efflorescence": [
        ("anti_salt", "Антисоль (очиститель высолов)", "л", 0.3, 2200),
        ("hydrophobizer", "Гидрофобизатор фасадный", "л", 0.25, 3800),
    ],
    "moss": [
        ("antiseptic", "Антисептик фасадный (биозащита)", "л", 0.3, 1800),
        ("primer_deep", "Грунтовка глубокого проникновения", "л", 0.15, 1200),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "rust": [
        ("rust_converter", "Преобразователь ржавчины", "л", 0.15, 2500),
        ("anticorrosion_primer", "Грунт антикоррозийный", "л", 0.12, 3200),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "spalling": [
        ("repair_compound", "Ремонтный состав (цементный)", "кг", 2.0, 1500),
        ("reinforcing_mesh", "Армирующая сетка фасадная", "м²", 1.1, 650),
        ("primer_deep", "Грунтовка глубокого проникновения", "л", 0.2, 1200),
        ("facade_putty", "Шпатлёвка фасадная", "кг", 0.8, 850),
        ("paint_facade", "Краска фасадная", "л", 0.25, 3500),
    ],
    "broken_glass": [
        ("glass_unit", "Стеклопакет двухкамерный", "шт", 1.0, 28000),
        ("sealant", "Герметик силиконовый", "шт", 0.5, 2800),
    ],
    "damaged_wood": [
        ("wood_antiseptic", "Антисептик для дерева", "л", 0.15, 2200),
        ("wood_putty", "Шпатлёвка по дереву", "кг", 0.5, 1200),
        ("wood_paint", "Краска/лак по дереву", "л", 0.15, 4500),
    ],
    "rusty_metal": [
        ("rust_converter", "Преобразователь ржавчины", "л", 0.2, 2500),
        ("anticorrosion_primer", "Грунт антикоррозийный", "л", 0.15, 3200),
        ("metal_paint", "Краска по металлу", "л", 0.12, 5200),
    ],
    "damaged_railing": [
        ("rust_converter", "Преобразователь ржавчины", "л", 0.2, 2500),
        ("anticorrosion_primer", "Грунт антикоррозийный", "л", 0.15, 3200),
        ("metal_paint", "Краска по металлу", "л", 0.12, 5200),
        ("welding_materials", "Сварочные материалы", "комп", 0.1, 15000),
    ],
}

# Labor rates per defect type: (display_name, unit, kzt_per_unit, norm_hours_per_unit)
LABOR_RATES = {
    "crack_surface": ("Заделка поверхностных трещин", "м.п.", 1500, 0.5),
    "crack_deep": ("Ремонт глубоких трещин с армированием", "м.п.", 3500, 1.5),
    "peeling": ("Восстановление штукатурного слоя", "м²", 4500, 2.0),
    "exposed_brick": ("Оштукатуривание оголённой кладки", "м²", 5000, 2.5),
    "water_damage": ("Устранение последствий протечек", "м²", 4000, 1.8),
    "efflorescence": ("Удаление высолов и гидрофобизация", "м²", 2000, 0.8),
    "moss": ("Биоочистка и антисептирование", "м²", 1800, 0.7),
    "rust": ("Антикоррозийная обработка", "м²", 2500, 1.0),
    "spalling": ("Восстановление бетонного покрытия", "м²", 5500, 2.5),
    "broken_glass": ("Замена стеклопакетов", "шт", 8000, 2.0),
    "damaged_wood": ("Ремонт деревянных элементов", "м²", 3500, 1.5),
    "rusty_metal": ("Антикоррозийная обработка металла", "м²", 3000, 1.2),
    "damaged_railing": ("Ремонт ограждений балконов", "м.п.", 5000, 2.0),
}

# Scaffolding cost per floor
SCAFFOLDING_COST_PER_FLOOR = 85000  # KZT per floor for full facade access
SCAFFOLDING_SETUP_HOURS = 8  # per floor

# Tax
VAT_RATE = 0.12  # 12% НДС in Kazakhstan
MATERIAL_SURPLUS = 0.10  # 10% запас материалов


def px_to_m2(area_px: int, total_px: int, total_m2: float = 450.0) -> float:
    """Convert pixel area to square meters using proportional scale."""
    if total_px == 0:
        return 0.0
    return (area_px / total_px) * total_m2


def estimate_floors(total_area_m2: float, avg_floor_height_m: float = 3.0,
                     avg_floor_width_m: float = 20.0) -> int:
    """Rough estimate of building floors from facade area."""
    if avg_floor_width_m == 0:
        return 1
    facade_height = total_area_m2 / avg_floor_width_m
    return max(1, int(math.ceil(facade_height / avg_floor_height_m)))


class RepairCalculator:
    """Calculate repair materials, labor, and total costs."""

    def __init__(self, total_area_m2: float = 450.0):
        self.total_area_m2 = total_area_m2

    def calculate(
        self,
        damages: List[dict],
        total_area_px: int,
        layer_analysis: Dict[str, dict],
    ) -> dict:
        """
        Calculate full repair estimate.

        Args:
            damages: list of damage dicts from FacadeAnalyzer
            total_area_px: total building area in pixels
            layer_analysis: wall layer analysis from FacadeAnalyzer

        Returns:
            dict with materials, labor, scaffolding, totals
        """
        materials_agg: Dict[str, RepairMaterial] = {}
        labor_items: List[RepairWork] = []

        for damage in damages:
            dtype = damage["type"]
            area_px = damage["area_px"]
            area_m2 = px_to_m2(area_px, total_area_px, self.total_area_m2)

            if area_m2 < 0.01:
                continue

            # Determine effective defect type (crack depth matters)
            effective_type = dtype
            if dtype == "crack":
                crack_depth = damage.get("crack_depth") or \
                    layer_analysis.get("crack", {}).get("crack_depth", "surface")
                effective_type = f"crack_{crack_depth}"

            # Materials
            rates = MATERIAL_RATES.get(effective_type, [])
            for mat_id, mat_display, unit, rate, price in rates:
                quantity = rate * area_m2
                if mat_id in materials_agg:
                    materials_agg[mat_id].quantity += quantity
                else:
                    materials_agg[mat_id] = RepairMaterial(
                        name=mat_id,
                        name_display=mat_display,
                        unit=unit,
                        price_per_unit=price,
                        quantity=quantity,
                    )

            # Labor
            if effective_type in LABOR_RATES:
                l_display, l_unit, l_price, l_hours = LABOR_RATES[effective_type]
                labor_items.append(RepairWork(
                    name=effective_type,
                    name_display=l_display,
                    unit=l_unit,
                    price_per_unit=l_price,
                    quantity=round(area_m2, 1),
                    norm_hours_per_unit=l_hours,
                ))

        # Apply surplus to materials
        for mat in materials_agg.values():
            mat.quantity = round(mat.quantity * (1 + MATERIAL_SURPLUS), 2)

        # Scaffolding
        floors = estimate_floors(self.total_area_m2)
        scaffolding_cost = floors * SCAFFOLDING_COST_PER_FLOOR
        scaffolding_hours = floors * SCAFFOLDING_SETUP_HOURS

        # Totals
        materials_total = sum(m.total_cost for m in materials_agg.values())
        labor_total = sum(l.total_cost for l in labor_items)
        total_hours = sum(l.total_hours for l in labor_items) + scaffolding_hours
        subtotal = materials_total + labor_total + scaffolding_cost
        vat = subtotal * VAT_RATE
        grand_total = subtotal + vat

        return {
            "materials": [
                {
                    "name": m.name,
                    "name_display": m.name_display,
                    "unit": m.unit,
                    "quantity": round(m.quantity, 2),
                    "price_per_unit": m.price_per_unit,
                    "total_cost": round(m.total_cost, 0),
                }
                for m in sorted(materials_agg.values(), key=lambda x: x.total_cost, reverse=True)
            ],
            "labor": [
                {
                    "name": l.name,
                    "name_display": l.name_display,
                    "unit": l.unit,
                    "quantity": l.quantity,
                    "price_per_unit": l.price_per_unit,
                    "total_cost": round(l.total_cost, 0),
                    "norm_hours": round(l.total_hours, 1),
                }
                for l in sorted(labor_items, key=lambda x: x.total_cost, reverse=True)
            ],
            "scaffolding": {
                "floors": floors,
                "cost": scaffolding_cost,
                "setup_hours": scaffolding_hours,
            },
            "summary": {
                "materials_total": round(materials_total, 0),
                "labor_total": round(labor_total, 0),
                "scaffolding_total": scaffolding_cost,
                "subtotal": round(subtotal, 0),
                "vat_rate": VAT_RATE,
                "vat_amount": round(vat, 0),
                "grand_total": round(grand_total, 0),
                "total_work_hours": round(total_hours, 1),
                "estimated_days": max(1, int(math.ceil(total_hours / 8))),
                "currency": "₸",
            },
            "costs_for_flutter": _build_flutter_costs(materials_agg, labor_items, scaffolding_cost, vat),
        }


def _build_flutter_costs(
    materials: Dict[str, RepairMaterial],
    labor: List[RepairWork],
    scaffolding: float,
    vat: float,
) -> List[dict]:
    """Build cost items in format compatible with Flutter CostBreakdownCard."""
    items = []

    # Group materials cost
    mat_total = sum(m.total_cost for m in materials.values())
    if mat_total > 0:
        items.append({
            "category": "Строительные материалы",
            "description": f"{len(materials)} наименований с учётом запаса 10%",
            "cost": round(mat_total, 0),
            "unit": "₸",
        })

    # Individual labor items
    for l in sorted(labor, key=lambda x: x.total_cost, reverse=True):
        if l.total_cost > 0:
            items.append({
                "category": l.name_display,
                "description": f"{l.quantity} {l.unit}",
                "cost": round(l.total_cost, 0),
                "unit": "₸",
            })

    # Scaffolding
    if scaffolding > 0:
        items.append({
            "category": "Леса и оборудование",
            "description": "Монтаж/демонтаж строительных лесов",
            "cost": round(scaffolding, 0),
            "unit": "₸",
        })

    # VAT
    if vat > 0:
        items.append({
            "category": "НДС (12%)",
            "description": "Налог на добавленную стоимость",
            "cost": round(vat, 0),
            "unit": "₸",
        })

    return items
