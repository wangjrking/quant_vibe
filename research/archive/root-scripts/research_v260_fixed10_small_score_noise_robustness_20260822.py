from __future__ import annotations

import copy
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
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_small_score_noise_robustness_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
SIGMAS = (0.00025, 0.00050, 0.00100)
SEEDS = tuple(range(10))


def metrics(context, policy: dict) -> dict:
    daily, actions = noise.run(context, policy)
    return round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )


def summarize(rows: list[dict], baseline: dict) -> dict:
    output = {"runs": int(len(rows))}
    for metric in (
        "cumulative_return", "cagr", "sharpe", "max_drawdown",
        "turnover_annualized", "average_invested_ratio",
    ):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[metric] = {
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "median_delta_vs_unperturbed": float(np.median(values) - baseline[metric]),
        }
    output["all_years_positive_fraction"] = float(np.mean([
        min(row["annual_returns"].values()) > 0.0 for row in rows
    ]))
    output["exactly10_fraction"] = float(np.mean([
        row["full_10_position_ratio"] == 1.0 for row in rows
    ]))
    return output


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered small-noise robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    baseline = metrics(context, policy)
    expected = checkpoint["current_best_equalweight"]
    baseline_equivalent = all(
        np.isclose(baseline[key], expected[key], rtol=0.0, atol=1e-12)
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    )
    if not baseline_equivalent:
        raise RuntimeError("small-noise baseline drifted")

    raw = {}
    summaries = {}
    for sigma in SIGMAS:
        rows = []
        for seed in SEEDS:
            case = noise.perturbed_context(context, sigma, seed)
            rows.append({"seed": seed, **metrics(case, policy)})
        key = f"{sigma:.5f}"
        raw[key] = rows
        summaries[key] = summarize(rows, baseline)

    repeat = metrics(noise.perturbed_context(context, 0.00050, 7), policy)
    deterministic = all(
        np.isclose(repeat[key], raw["0.00050"][7][key], rtol=0.0, atol=1e-12)
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    )
    if not deterministic:
        raise RuntimeError("small-noise deterministic replay failed")

    result = {
        "status": "small_score_noise_robustness_complete_2026_not_opened",
        "baseline": baseline,
        "sigma_results": summaries,
        "raw_runs": raw,
        "baseline_equivalent": baseline_equivalent,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "small_score_noise_robustness.json", result)
    print(json.dumps({
        "status": result["status"],
        "sigma_results": summaries,
        "deterministic_replay": deterministic,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
