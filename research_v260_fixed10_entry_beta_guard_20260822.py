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

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_renewal_maintenance_20260822 as renewal
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_entry_beta_guard_20260822"
)
CANDIDATE_BETA_CAPS = {
    "production_entry": None,
    "entry_beta_at_most_200": 2.00,
    "entry_beta_at_most_150": 1.50,
}
ROBUSTNESS_BETA_CAPS = {
    "control_entry_beta_at_most_175": 1.75,
    "control_entry_beta_at_most_225": 2.25,
}
BETA_LOOKBACK = 60
BETA_MIN_OBSERVATIONS = 40
MIN_CAGR_IMPROVEMENT = 0.005
MIN_SHARPE_IMPROVEMENT = 0.01
MIN_DRAWDOWN_IMPROVEMENT = 0.005


def trailing_market_beta(
    close_qfq: np.ndarray,
    lookback: int,
    min_observations: int,
) -> np.ndarray:
    close = np.asarray(close_qfq, dtype=np.float64)
    if close.ndim != 2 or lookback < 2 or min_observations < 2:
        raise ValueError("rolling beta configuration is invalid")
    returns = np.full(close.shape, np.nan, dtype=np.float64)
    valid_prices = (
        np.isfinite(close[1:])
        & np.isfinite(close[:-1])
        & (close[1:] > 0)
        & (close[:-1] > 0)
    )
    adjacent = np.full_like(close[1:], np.nan)
    adjacent[valid_prices] = close[1:][valid_prices] / close[:-1][valid_prices] - 1.0
    returns[1:] = adjacent
    market = np.full(len(returns), np.nan, dtype=np.float64)
    has_cross_section = np.any(np.isfinite(returns[1:]), axis=1)
    market_values = market[1:]
    market_values[has_cross_section] = np.nanmedian(
        returns[1:][has_cross_section], axis=1
    )
    valid = np.isfinite(returns) & np.isfinite(market[:, None])
    x = np.where(valid, returns, 0.0)
    y = np.where(valid, market[:, None], 0.0)

    def cumulative(values: np.ndarray) -> np.ndarray:
        return np.vstack(
            [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)]
        )

    sum_x = cumulative(x)
    sum_y = cumulative(y)
    sum_xy = cumulative(x * y)
    sum_y2 = cumulative(y * y)
    count = np.vstack(
        [
            np.zeros((1, close.shape[1]), dtype=np.int32),
            np.cumsum(valid, axis=0, dtype=np.int32),
        ]
    )
    result = np.full(close.shape, np.nan, dtype=np.float64)
    for end in range(1, len(close)):
        start = max(0, end - int(lookback) + 1)
        n = count[end + 1] - count[start]
        sx = sum_x[end + 1] - sum_x[start]
        sy = sum_y[end + 1] - sum_y[start]
        sxy = sum_xy[end + 1] - sum_xy[start]
        sy2 = sum_y2[end + 1] - sum_y2[start]
        covariance = sxy - np.divide(
            sx * sy, n, out=np.zeros_like(sx), where=n > 0
        )
        market_variance = sy2 - np.divide(
            sy * sy, n, out=np.zeros_like(sy), where=n > 0
        )
        usable = (n >= int(min_observations)) & (market_variance > 1e-16)
        result[end, usable] = covariance[usable] / market_variance[usable]
    return result.astype(np.float32)


def entry_selection_mask(
    candidate_id: str,
    production_mask: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    all_caps = {**CANDIDATE_BETA_CAPS, **ROBUSTNESS_BETA_CAPS}
    if candidate_id not in all_caps:
        raise ValueError(f"unknown candidate: {candidate_id}")
    result = np.asarray(production_mask, dtype=np.bool_).copy()
    cap = all_caps[candidate_id]
    if cap is not None:
        values = np.asarray(beta, dtype=np.float64)
        result &= ~np.isfinite(values) | (values <= float(cap))
    return result


def materially_better(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["cagr"] - baseline["cagr"] >= MIN_CAGR_IMPROVEMENT
        or candidate["sharpe"] - baseline["sharpe"] >= MIN_SHARPE_IMPROVEMENT
        or baseline["max_drawdown"] - candidate["max_drawdown"]
        >= MIN_DRAWDOWN_IMPROVEMENT
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered beta-guard development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    production_mask = harness.v174.selection_mask(
        arrays, definition["max_rank_deterioration"]
    )
    beta = trailing_market_beta(
        arrays["close_qfq"], BETA_LOOKBACK, BETA_MIN_OBSERVATIONS
    )
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    entry_block = np.zeros(score.shape, dtype=np.bool_)
    maintenance_block = quality.maintenance_quality_block(
        score, policy.get("maintenance_topup_requires_score") is not None
    )
    results = {}
    run_cache = {}
    mask_cache = {}

    all_caps = {**CANDIDATE_BETA_CAPS, **ROBUSTNESS_BETA_CAPS}
    for candidate_id, beta_cap in all_caps.items():
        selection_mask = entry_selection_mask(candidate_id, production_mask, beta)
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
            maintenance_buy_block_mask_override=maintenance_block,
            selection_mask_override=selection_mask,
        )
        daily, actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            policy,
            round1.DEVELOPMENT_END,
            slip=round1.BASELINE_COST,
            record_actions=True,
            **common,
        )
        stress_daily, stress_actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            policy,
            round1.DEVELOPMENT_END,
            slip=round1.STRESS_COST,
            record_actions=True,
            **common,
        )
        train = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        )
        holdout = round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        )
        windows = {
            str(offset): round1.evaluate_run(
                daily,
                actions,
                robustness.window_start_date(arrays, offset),
                round1.DEVELOPMENT_END,
            )
            for offset in (5, 20, 60)
        }
        results[candidate_id] = {
            "policy": {
                **policy,
                "new_entry_market_beta_max": beta_cap,
                "market_beta_lookback": BETA_LOOKBACK,
                "market_beta_min_observations": BETA_MIN_OBSERVATIONS,
            },
            "metrics_0_30pct": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            ),
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout,
            "window_start_metrics": windows,
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "selection_mask_true_count": int(selection_mask.sum()),
        }
        run_cache[candidate_id] = (daily, actions)
        mask_cache[candidate_id] = selection_mask

    baseline_id = "production_entry"
    current = results[baseline_id]["metrics_0_30pct"]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(current[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("beta-guard baseline drifted from checkpoint")

    ranked_id = max(
        CANDIDATE_BETA_CAPS,
        key=lambda key: renewal.selection_key(results[key]),
    )
    baseline_train = results[baseline_id]["train_2022_2024"]
    neighborhood_robust = bool(
        all(
            results[control_id]["train_2022_2024"]["cagr"]
            >= baseline_train["cagr"] - 0.005
            and results[control_id]["train_2022_2024"]["sharpe"]
            >= baseline_train["sharpe"] - 0.01
            and results[control_id]["train_2022_2024"]["max_drawdown"]
            <= baseline_train["max_drawdown"] + 0.01
            for control_id in ROBUSTNESS_BETA_CAPS
        )
    )
    material_improvement = bool(
        ranked_id != baseline_id
        and materially_better(
            results[ranked_id]["train_2022_2024"],
            baseline_train,
        )
        and neighborhood_robust
    )
    selected_id = ranked_id if material_improvement else baseline_id
    selected_daily, selected_actions = run_cache[selected_id]
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=entry_block,
        maintenance_buy_block_mask_override=maintenance_block,
        selection_mask_override=mask_cache[selected_id],
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("beta-guard deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATE_BETA_CAPS),
        "predeclared_robustness_controls_not_selectable": list(
            ROBUSTNESS_BETA_CAPS
        ),
        "only_change": (
            "skip only new entries whose trailing PIT market beta exceeds a coarse "
            "risk cap; preserve the production score, exits, full investment, and refill"
        ),
        "results": results,
        "walk_forward_selection_evidence": robustness.walk_forward_selections(results),
        "selected_candidate_id": selected_id,
        "selected_reason": (
            "material aggregate 2022-2024 improvement"
            if material_improvement
            else "candidate did not clear the predeclared practical improvement gate"
        ),
        "practical_improvement_gate": {
            "minimum_cagr_improvement": MIN_CAGR_IMPROVEMENT,
            "minimum_sharpe_improvement": MIN_SHARPE_IMPROVEMENT,
            "minimum_drawdown_improvement": MIN_DRAWDOWN_IMPROVEMENT,
            "raw_ranked_candidate_id": ranked_id,
            "neighborhood_robust": neighborhood_robust,
            "passed": material_improvement,
        },
        "selected_policy": copy.deepcopy(results[selected_id]["policy"]),
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "selected": selected_id,
                "ranked": ranked_id,
                "practical_gate": result["practical_improvement_gate"],
                "results": {
                    key: {
                        "train": value["train_2022_2024"],
                        "holdout": value["holdout_2025_not_used_for_selection"],
                    }
                    for key, value in results.items()
                },
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
