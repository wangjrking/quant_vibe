from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_score_noise_robustness_20260822 as noise


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_score_noise_rank_calibration_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
SIGMAS = (0.00025, 0.00050, 0.00100, 0.00250)
SEEDS = tuple(range(10))
TOP_K = 10


def finite_order(score_row: np.ndarray, order_row: np.ndarray) -> np.ndarray:
    ranked = np.asarray(order_row, dtype=np.int64)
    return ranked[np.isfinite(score_row[ranked])]


def natural_gap_summary(score: np.ndarray, order: np.ndarray) -> dict:
    top10_11 = []
    adjacent_top20 = []
    universe_sizes = []
    for score_row, order_row in zip(score, order):
        ranked = finite_order(score_row, order_row)
        universe_sizes.append(int(ranked.size))
        if ranked.size > TOP_K:
            top10_11.append(float(score_row[ranked[TOP_K - 1]] - score_row[ranked[TOP_K]]))
        limit = min(20, ranked.size)
        if limit > 1:
            values = score_row[ranked[:limit]]
            adjacent_top20.extend((values[:-1] - values[1:]).tolist())
    return {
        "sessions": int(len(universe_sizes)),
        "median_finite_universe": float(np.median(universe_sizes)),
        "top10_vs_11_gap": quantiles(top10_11),
        "adjacent_gap_within_top20": quantiles(adjacent_top20),
    }


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def compare_orders(
    score: np.ndarray,
    baseline_order: np.ndarray,
    perturbed_order: np.ndarray,
) -> dict:
    overlaps = []
    displaced = []
    max_displaced = []
    unchanged = []
    for score_row, baseline_row, perturbed_row in zip(
        score, baseline_order, perturbed_order
    ):
        baseline = finite_order(score_row, baseline_row)
        perturbed = finite_order(score_row, perturbed_row)
        if baseline.size < TOP_K or perturbed.size < TOP_K:
            continue
        baseline_top = baseline[:TOP_K]
        perturbed_top = perturbed[:TOP_K]
        overlap = len(set(baseline_top.tolist()) & set(perturbed_top.tolist()))
        positions = np.empty(score_row.size, dtype=np.int64)
        positions[perturbed] = np.arange(perturbed.size, dtype=np.int64)
        moves = np.abs(positions[baseline_top] - np.arange(TOP_K, dtype=np.int64))
        overlaps.append(overlap / TOP_K)
        displaced.append(float(np.mean(moves)))
        max_displaced.append(int(np.max(moves)))
        unchanged.append(int(np.array_equal(baseline_top, perturbed_top)))
    return {
        "sessions": int(len(overlaps)),
        "mean_top10_overlap": float(np.mean(overlaps)),
        "median_top10_overlap": float(np.median(overlaps)),
        "p10_top10_overlap": float(np.quantile(overlaps, 0.10)),
        "mean_absolute_rank_displacement_of_original_top10": float(np.mean(displaced)),
        "p90_max_rank_displacement_of_original_top10": float(
            np.quantile(max_displaced, 0.90)
        ),
        "exact_top10_order_unchanged_fraction": float(np.mean(unchanged)),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = [key for key in rows[0] if key != "sessions"]
    return {
        "runs": int(len(rows)),
        **{
            key: {
                "median": float(np.median([row[key] for row in rows])),
                "p10": float(np.quantile([row[key] for row in rows], 0.10)),
                "p90": float(np.quantile([row[key] for row in rows], 0.90)),
            }
            for key in keys
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered rank-noise calibration")
    context = harness.load_context(checkpoint["selected_policy"])
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("unexpected data boundary")

    raw = {}
    summary = {}
    for sigma in SIGMAS:
        rows = []
        for seed in SEEDS:
            perturbed = noise.perturbed_context(context, sigma, seed)
            rows.append(compare_orders(context.score, context.order, perturbed.order))
        key = f"{sigma:.5f}"
        raw[key] = rows
        summary[key] = aggregate(rows)

    result = {
        "status": "score_noise_rank_calibration_complete_2026_not_opened",
        "natural_score_spacing": natural_gap_summary(context.score, context.order),
        "noise_rank_effect": summary,
        "raw_runs": raw,
        "interpretation": (
            "Noise magnitude must be interpreted relative to daily cross-sectional "
            "score spacing; this diagnostic does not select or alter a strategy rule."
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "score_noise_rank_calibration.json", result)
    print(json.dumps({
        "status": result["status"],
        "natural_score_spacing": result["natural_score_spacing"],
        "noise_rank_effect": result["noise_rank_effect"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
