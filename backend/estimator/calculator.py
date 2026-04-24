"""
The calculator turns a list of defects (already in real-world metres, thanks
to the calibration step) into a full RUB estimate.

Flow per defect:
    defect → recipe (see layer_recipes.py)
           → for each step:
               - labor cost  = labor_price_rub × defect_area × waste_factor_labor
               - material[i] = mat_price_rub × mat_rate × defect_area × waste_factor_materials
    Sum → subtotal → + VAT → grand total.

The scaffolding heuristic from the old calculator is kept (same formula),
translated into RUB via a separate "labor" key.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any

from backend.calibration import ScaleCalibration
from backend.core.config import get_settings
from backend.estimator.layer_recipes import recipe_for
from backend.estimator.prices import PriceBook, load_price_book


# ── sizing helpers ────────────────────────────────────────────────────────

def _defect_size(
    damage: dict,
    layer_info: dict | None,
    step_unit: str,
    scale: ScaleCalibration,
) -> float:
    """
    Return the quantity to charge a step against, in the step's unit.

    Cracks are priced per linear metre; everything else per m².
    """
    area_px = float(damage.get("area_px", 0))
    area_m2 = float(damage.get("area_m2") or scale.area_px_to_m2(area_px))

    if step_unit == "m":
        # Approximate linear length as area / typical_width. A crack is about
        # 5mm wide; this is a rough but stable rule of thumb.
        approx_length = area_m2 / 0.005 if area_m2 > 0 else 0
        return round(approx_length, 2)

    if step_unit == "unit":
        # For broken_glass / glass_crack: count panes based on area.
        return max(1.0, round(area_m2 / 1.2, 1))

    return round(area_m2, 2)


def _estimate_floors(area_m2: float) -> int:
    """Tiny heuristic: assume 20m facade width, 3m floor height."""
    if area_m2 <= 0:
        return 1
    facade_h = area_m2 / 20.0
    return max(1, int(math.ceil(facade_h / 3.0)))


def _scaffolding_line(
    total_area_m2: float, price_book: PriceBook
) -> dict[str, Any]:
    floors = _estimate_floors(total_area_m2)
    # Price per floor (labour-only cost in RUB; mat cost is rental).
    per_floor, _unit = price_book.labor_price("scaffolding_per_floor") or (18000.0, "floor")
    cost = floors * per_floor
    return {
        "key": "scaffolding",
        "display": "Леса строительные",
        "unit": "floor",
        "quantity": floors,
        "price_per_unit": per_floor,
        "total_cost": round(cost, 0),
        "norm_hours": 8 * floors,
    }


# ── public entrypoint ─────────────────────────────────────────────────────


async def build_estimate_from_analysis(
    damages: list[dict],
    layer_analysis: dict[str, dict],
    scale: ScaleCalibration,
    waste_factor: float | None = None,
    vat_rate: float | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    waste = waste_factor if waste_factor is not None else settings.waste_factor
    vat = vat_rate if vat_rate is not None else settings.vat_rate

    book = await load_price_book()

    # Aggregate materials / labor across all defects.
    material_agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"quantity": 0.0, "total_cost": 0.0}
    )
    labor_items: list[dict[str, Any]] = []
    total_area_m2 = 0.0

    for damage in damages:
        dtype = damage["type"]
        area_m2 = float(damage.get("area_m2", 0) or 0)
        if area_m2 <= 0:
            area_m2 = scale.area_px_to_m2(damage.get("area_px", 0))
        total_area_m2 += area_m2
        if area_m2 < 0.01 and dtype not in {"broken_glass", "glass_crack"}:
            continue

        crack_depth = damage.get("crack_depth") or (
            layer_analysis.get("crack", {}).get("crack_depth")
        )
        steps = recipe_for(dtype, crack_depth)

        for step in steps:
            qty = _defect_size(damage, layer_analysis.get(dtype), step.unit, scale)

            # Labour
            lp = book.labor_price(step.labor_key) or (0.0, step.unit)
            labor_price, _lu = lp
            labor_cost = labor_price * qty * waste
            labor_items.append(
                {
                    "key": step.labor_key,
                    "display": step.labor_display_ru,
                    "unit": step.unit,
                    "quantity": qty,
                    "price_per_unit": labor_price,
                    "total_cost": round(labor_cost, 0),
                    "norm_hours": round(step.norm_hours * qty, 1),
                    "defect_type": dtype,
                }
            )

            # Materials
            for mc in step.materials:
                mat_qty = mc.rate * qty * waste
                mp = book.material_price(mc.material_key)
                if mp is None:
                    mat_price, mat_unit = 0.0, mc.unit
                else:
                    mat_price, mat_unit = mp
                cost = mat_price * mat_qty
                agg = material_agg[mc.material_key]
                if agg["total_cost"] == 0:
                    agg.update(
                        {
                            "display": mc.display_ru,
                            "unit": mat_unit,
                            "price_per_unit": mat_price,
                        }
                    )
                agg["quantity"] = round(
                    float(agg["quantity"]) + mat_qty, 2
                )
                agg["total_cost"] = round(
                    float(agg["total_cost"]) + cost, 0
                )

    # Build ordered lists
    materials_list = [
        {"key": k, **v}
        for k, v in sorted(
            material_agg.items(), key=lambda kv: kv[1]["total_cost"], reverse=True
        )
    ]
    labor_list = sorted(labor_items, key=lambda x: x["total_cost"], reverse=True)

    # Scaffolding (based on damaged area as proxy for facade size)
    scaffolding = _scaffolding_line(max(total_area_m2, 1.0), book)

    # Totals
    materials_total = sum(m["total_cost"] for m in materials_list)
    labor_total = sum(l["total_cost"] for l in labor_list)
    subtotal = materials_total + labor_total + scaffolding["total_cost"]
    vat_amount = subtotal * vat
    grand_total = subtotal + vat_amount
    total_hours = sum(l["norm_hours"] for l in labor_list) + scaffolding["norm_hours"]

    repair_estimate: dict[str, Any] = {
        "currency": "RUB",
        "currency_symbol": "₽",
        "materials": materials_list,
        "labor": labor_list,
        "scaffolding": scaffolding,
        "summary": {
            "materials_total": round(materials_total, 0),
            "labor_total": round(labor_total, 0),
            "scaffolding_total": scaffolding["total_cost"],
            "subtotal": round(subtotal, 0),
            "vat_rate": vat,
            "vat_amount": round(vat_amount, 0),
            "grand_total": round(grand_total, 0),
            "waste_factor": waste,
            "total_hours": round(total_hours, 1),
            "estimated_days": max(1, int(math.ceil(total_hours / 8))),
        },
        # Flutter legacy structure (see lib/models/analysis_result.dart CostItem list)
        "costs_for_flutter": _flutter_costs(
            materials_list, labor_list, scaffolding, vat_amount
        ),
    }

    return {
        "repair_estimate": repair_estimate,
        "price_snapshot_date": (
            book.snapshot_date.isoformat() if book.snapshot_date else None
        ),
        "price_source": book.source,
        "stale": book.stale,
    }


# ── Flutter shim ──────────────────────────────────────────────────────────


def _flutter_costs(
    materials_list: list[dict],
    labor_list: list[dict],
    scaffolding: dict,
    vat_amount: float,
) -> list[dict]:
    items: list[dict] = []
    mat_total = sum(m["total_cost"] for m in materials_list)
    if mat_total > 0:
        items.append(
            {
                "category": "Строительные материалы",
                "description": f"{len(materials_list)} наименований с учётом запаса",
                "cost": round(mat_total, 0),
                "unit": "₽",
            }
        )
    for l in labor_list[:15]:  # keep breakdown digestible
        items.append(
            {
                "category": l["display"],
                "description": f"{l['quantity']} {l['unit']}",
                "cost": l["total_cost"],
                "unit": "₽",
            }
        )
    if scaffolding["total_cost"] > 0:
        items.append(
            {
                "category": "Леса и оборудование",
                "description": f"Монтаж/демонтаж — {scaffolding['quantity']} эт.",
                "cost": scaffolding["total_cost"],
                "unit": "₽",
            }
        )
    if vat_amount > 0:
        items.append(
            {
                "category": "НДС 20%",
                "description": "Налог на добавленную стоимость",
                "cost": round(vat_amount, 0),
                "unit": "₽",
            }
        )
    return items
