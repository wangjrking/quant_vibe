from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import run_open_gap_frequency_probe_20260630 as base


REPORT_DIR = base.REPORT_DIR
base.OUT_SIGNAL_DIR = REPORT_DIR / "falling_knife_dayrisk_signals"
base.OUT_LOG_DIR = REPORT_DIR / "falling_knife_dayrisk_logs"
base.OUT_CSV = REPORT_DIR / "falling_knife_dayrisk_probe_20260630.csv"
base.OUT_JSON = REPORT_DIR / "falling_knife_dayrisk_probe_20260630.json"


RISK_CASES: list[dict[str, Any]] = [
    {"name": "fall_avg_pct_m4_scale50", "mode": "day_scale", "avg_pct_lte": -4.0, "scale": 0.50},
    {"name": "fall_avg_pct_m5_scale35", "mode": "day_scale", "avg_pct_lte": -5.0, "scale": 0.35},
    {"name": "fall_avg_2d_m6_scale50", "mode": "day_scale", "avg_two_day_lte": -0.06, "scale": 0.50},
    {"name": "fall_avg_2d_m8_scale35", "mode": "day_scale", "avg_two_day_lte": -0.08, "scale": 0.35},
    {"name": "fall_row_pct_m6_scale50", "mode": "row_scale", "pct_lte": -6.0, "scale": 0.50},
    {"name": "fall_row_2d_m8_scale50", "mode": "row_scale", "two_day_lte": -0.08, "scale": 0.50},
    {"name": "fall_hybrid_day_scale35", "mode": "hybrid_day_scale", "avg_pct_lte": -4.0, "avg_two_day_lte": -0.06, "scale": 0.35},
    {"name": "fall_hybrid_row_scale35", "mode": "hybrid_row_scale", "pct_lte": -6.0, "two_day_lte": -0.08, "scale": 0.35},
    {"name": "fall_skip_day_deep", "mode": "day_skip", "avg_pct_lte": -6.0, "avg_two_day_lte": -0.08},
    {"name": "fall_tier_row", "mode": "tier_row"},
]

base.OPEN_GAP_CASES = RISK_CASES


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _day_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("signal_date") or "")].append(row)
    stats: dict[str, dict[str, float]] = {}
    for signal_date, items in grouped.items():
        pct_values = [_float(row.get("pct_chg")) for row in items]
        two_day_values = [_float(row.get("two_day_ret")) for row in items]
        pct_values = [value for value in pct_values if value is not None]
        two_day_values = [value for value in two_day_values if value is not None]
        stats[signal_date] = {
            "avg_pct": sum(pct_values) / len(pct_values) if pct_values else 0.0,
            "avg_two_day": sum(two_day_values) / len(two_day_values) if two_day_values else 0.0,
        }
    return stats


def _risk_triggered(case: dict[str, Any], stats: dict[str, float]) -> bool:
    avg_pct_lte = case.get("avg_pct_lte")
    avg_two_day_lte = case.get("avg_two_day_lte")
    return (
        avg_pct_lte is not None
        and stats["avg_pct"] <= float(avg_pct_lte)
    ) or (
        avg_two_day_lte is not None
        and stats["avg_two_day"] <= float(avg_two_day_lte)
    )


def _row_triggered(case: dict[str, Any], row: dict[str, Any]) -> bool:
    pct = _float(row.get("pct_chg"))
    two_day = _float(row.get("two_day_ret"))
    pct_lte = case.get("pct_lte")
    two_day_lte = case.get("two_day_lte")
    return (
        pct_lte is not None
        and pct is not None
        and pct <= float(pct_lte)
    ) or (
        two_day_lte is not None
        and two_day is not None
        and two_day <= float(two_day_lte)
    )


def _tier_scale(row: dict[str, Any]) -> float:
    pct = _float(row.get("pct_chg"))
    two_day = _float(row.get("two_day_ret"))
    if two_day is not None and two_day <= -0.10:
        return 0.25
    if two_day is not None and two_day <= -0.08:
        return 0.35
    if two_day is not None and two_day <= -0.06:
        return 0.50
    if pct is not None and pct <= -8.0:
        return 0.50
    if pct is not None and pct <= -6.0:
        return 0.65
    return 1.0


def _signal_rows(
    probe_base: dict[str, Any],
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    gaps: dict[tuple[str, str], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del gaps
    stats_by_day = _day_stats(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("signal_date") or "")].append(row)

    out: list[dict[str, Any]] = []
    skipped = 0
    scaled = 0
    for signal_date, items in grouped.items():
        day_stats = stats_by_day[signal_date]
        if case["mode"] == "day_skip" and _risk_triggered(case, day_stats):
            skipped += len(items)
            continue
        for row in items:
            target = _float(row.get("target_pct")) or float(probe_base["target_position_pct"])
            scale = 1.0
            mode = case["mode"]
            if mode == "day_scale" and _risk_triggered(case, day_stats):
                scale = float(case["scale"])
            elif mode == "row_scale" and _row_triggered(case, row):
                scale = float(case["scale"])
            elif mode == "hybrid_day_scale" and _risk_triggered(case, day_stats):
                scale = float(case["scale"])
            elif mode == "hybrid_row_scale" and _row_triggered(case, row):
                scale = float(case["scale"])
            elif mode == "tier_row":
                scale = _tier_scale(row)
            elif mode not in {"day_scale", "row_scale", "hybrid_day_scale", "hybrid_row_scale", "day_skip", "tier_row"}:
                raise ValueError(f"unknown mode: {mode}")
            if scale != 1.0:
                scaled += 1
            item = dict(row)
            item["target_pct"] = f"{target * scale:.5f}"
            item["strategy_variant"] = f"{probe_base['base']}__{case['name']}"
            item["filter_name"] = str(case["name"])
            item["day_avg_pct_chg"] = f"{day_stats['avg_pct']:.6f}"
            item["day_avg_two_day_ret"] = f"{day_stats['avg_two_day']:.8f}"
            item["dayrisk_scale"] = f"{scale:.5f}"
            out.append(item)

    out.sort(key=lambda row: (str(row.get("signal_date") or ""), int(float(row.get("rank") or 9999)), str(row.get("stock_code") or "")))
    counts = Counter(row["signal_date"] for row in out)
    target = int(probe_base["max_positions"])
    return out, {
        "signal_rows": len(out),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < target),
        "empty_days_vs_base": 986 - len(counts),
        "skipped_rows": skipped,
        "scaled_rows": scaled,
        "missing_gap_rows": 0,
    }


base._signal_rows = _signal_rows


if __name__ == "__main__":
    base.main()
