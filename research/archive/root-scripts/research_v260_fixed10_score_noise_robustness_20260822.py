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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_score_noise_robustness_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
SIGMAS = (0.0025, 0.0050, 0.0100)
SEEDS = tuple(range(10))


def stable_descending_order(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    safe = np.where(np.isfinite(values), values, -np.inf)
    return np.argsort(-safe, axis=1, kind="stable").astype(np.int64)


def perturbed_context(context, sigma: float, seed: int):
    case = copy.copy(context)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), size=context.score.shape)
    finite = np.isfinite(context.score)
    score = np.full(context.score.shape, np.nan, dtype=np.float64)
    score[finite] = np.clip(context.score[finite] + noise[finite], 0.0, 1.0)
    case.score = score
    case.order = stable_descending_order(score)
    return case


def run(context, policy: dict):
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def summarize(rows: list[dict], baseline: dict) -> dict:
    output = {"runs": int(len(rows))}
    for metric in (
        "cumulative_return", "cagr", "sharpe", "max_drawdown",
        "turnover_annualized", "average_invested_ratio",
    ):
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        output[metric] = {
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "worst": float(values.min()),
            "best": float(values.max()),
            "median_delta_vs_unperturbed": float(np.median(values) - baseline[metric]),
        }
    output["all_calendar_years_positive_fraction"] = float(np.mean([
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
        raise PermissionError("2026 data entered score-noise robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    baseline_daily, baseline_actions = run(context, policy)
    baseline = round1.evaluate_run(
        baseline_daily,
        baseline_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    expected = checkpoint["current_best_equalweight"]
    equivalent = {
        key: bool(np.isclose(baseline[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalent.values()):
        raise RuntimeError("score-noise baseline drifted")
    results = {}
    raw_rows = {}
    for sigma in SIGMAS:
        key = f"{sigma:.4f}"
        rows = []
        for seed in SEEDS:
            case = perturbed_context(context, sigma, seed)
            daily, actions = run(case, policy)
            metrics = round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            )
            rows.append({"seed": seed, **metrics})
        raw_rows[key] = rows
        results[key] = summarize(rows, baseline)
    repeat_case = perturbed_context(context, 0.0050, 7)
    repeat_daily, repeat_actions = run(repeat_case, policy)
    reference = raw_rows["0.0050"][7]
    repeat_metrics = round1.evaluate_run(
        repeat_daily,
        repeat_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    deterministic = all(
        np.isclose(repeat_metrics[key], reference[key], rtol=0.0, atol=1e-12)
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    )
    if not deterministic:
        raise RuntimeError("score-noise replay failed")
    result = {
        "status": "score_noise_robustness_complete_2026_not_opened",
        "method": (
            "add deterministic zero-mean Gaussian noise to finite smoothed percentile "
            "scores, rebuild order, and replay without selecting any parameter"
        ),
        "unperturbed": baseline,
        "sigma_results": results,
        "raw_runs": raw_rows,
        "baseline_equivalence": equivalent,
        "deterministic_seed7_sigma005_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "score_noise_robustness.json", result)
    print(json.dumps({
        "status": result["status"],
        "sigma_results": results,
        "deterministic": deterministic,
        "validation_2026_opened": False,
    }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
