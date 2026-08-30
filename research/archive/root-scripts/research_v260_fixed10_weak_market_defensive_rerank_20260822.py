from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_renewal_maintenance_20260822 as renewal
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_weak_market_defensive_rerank_20260822"
)
CANDIDATES = (
    "production_order",
    "weak_market_low_vol_top30",
    "weak_market_rank1d70_rank10d30_top30",
)
VOLATILITY_WINDOW = 20
DEFENSIVE_POOL_SIZE = 30


def trailing_log_volatility(close_qfq: np.ndarray, window: int) -> np.ndarray:
    close = np.asarray(close_qfq, dtype=np.float64)
    if close.ndim != 2 or window < 2:
        raise ValueError("close matrix and rolling window are invalid")
    returns = np.full(close.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(close[1:])
        & np.isfinite(close[:-1])
        & (close[1:] > 0)
        & (close[:-1] > 0)
    )
    adjacent_returns = np.full_like(close[1:], np.nan)
    adjacent_returns[valid] = np.log(close[1:][valid] / close[:-1][valid])
    returns[1:] = adjacent_returns
    finite = np.isfinite(returns)
    values = np.where(finite, returns, 0.0)
    cumulative = np.vstack([np.zeros((1, close.shape[1])), np.cumsum(values, axis=0)])
    cumulative_sq = np.vstack(
        [np.zeros((1, close.shape[1])), np.cumsum(values * values, axis=0)]
    )
    counts = np.vstack(
        [np.zeros((1, close.shape[1]), dtype=np.int32), np.cumsum(finite, axis=0)]
    )
    result = np.full(close.shape, np.nan, dtype=np.float64)
    for end in range(window, close.shape[0]):
        start = end - window + 1
        count = counts[end + 1] - counts[start]
        total = cumulative[end + 1] - cumulative[start]
        total_sq = cumulative_sq[end + 1] - cumulative_sq[start]
        complete = count == window
        variance = np.maximum(total_sq / window - (total / window) ** 2, 0.0)
        result[end, complete] = np.sqrt(variance[complete])
    return result


def defensive_order(
    production_order: np.ndarray,
    volatility: np.ndarray,
    weak_market: np.ndarray,
    pool_size: int,
) -> np.ndarray:
    original = np.asarray(production_order)
    vol = np.asarray(volatility, dtype=np.float64)
    weak = np.asarray(weak_market, dtype=np.bool_)
    if original.shape != vol.shape or weak.shape != (original.shape[0],):
        raise ValueError("defensive rerank shapes do not align")
    result = original.copy()
    limit = min(max(int(pool_size), 1), original.shape[1])
    for day in np.flatnonzero(weak):
        head = original[day, :limit]
        head_position = np.arange(limit, dtype=np.int64)
        head_vol = vol[day, head]
        safe_vol = np.where(np.isfinite(head_vol), head_vol, np.inf)
        rearranged = np.lexsort((head_position, safe_vol))
        result[day, :limit] = head[rearranged]
    return result


def bounded_score_order(
    production_order: np.ndarray,
    alternative_score: np.ndarray,
    weak_market: np.ndarray,
    pool_size: int,
) -> np.ndarray:
    original = np.asarray(production_order)
    score = np.asarray(alternative_score, dtype=np.float64)
    weak = np.asarray(weak_market, dtype=np.bool_)
    if original.shape != score.shape or weak.shape != (original.shape[0],):
        raise ValueError("bounded score rerank shapes do not align")
    result = original.copy()
    limit = min(max(int(pool_size), 1), original.shape[1])
    for day in np.flatnonzero(weak):
        head = original[day, :limit]
        head_position = np.arange(limit, dtype=np.int64)
        head_score = score[day, head]
        safe_score = np.where(np.isfinite(head_score), head_score, -np.inf)
        rearranged = np.lexsort((head_position, -safe_score))
        result[day, :limit] = head[rearranged]
    return result


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
        raise PermissionError("2026 data entered defensive rerank development")
    definition = harness.production_definition(protocol)
    score, production_order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    volatility = trailing_log_volatility(arrays["close_qfq"], VOLATILITY_WINDOW)
    weak_market = ~regime.strong_market_mask(score, protocol)
    defensive = defensive_order(
        production_order, volatility, weak_market, DEFENSIVE_POOL_SIZE
    )
    short_term_blend = (
        0.70 * np.asarray(arrays["rank_1d"], dtype=np.float64)
        + 0.30 * np.asarray(arrays["rank_10d"], dtype=np.float64)
    )
    short_term_order = bounded_score_order(
        production_order, short_term_blend, weak_market, DEFENSIVE_POOL_SIZE
    )
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    entry_block = np.zeros(score.shape, dtype=np.bool_)
    maintenance_block = quality.maintenance_quality_block(score, True)

    results = {}
    run_cache = {}
    for candidate_id in CANDIDATES:
        current_order = {
            "production_order": production_order,
            "weak_market_low_vol_top30": defensive,
            "weak_market_rank1d70_rank10d30_top30": short_term_order,
        }[candidate_id]
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
            maintenance_buy_block_mask_override=maintenance_block,
        )
        daily, actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            current_order,
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
            current_order,
            policy,
            round1.DEVELOPMENT_END,
            slip=round1.STRESS_COST,
            record_actions=True,
            **common,
        )
        metrics = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        train = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        )
        holdout = round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        )
        stress = round1.evaluate_run(
            stress_daily,
            stress_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
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
                "weak_market_entry_order": {
                    "production_order": None,
                    "weak_market_low_vol_top30": {
                        "production_top_pool": DEFENSIVE_POOL_SIZE,
                        "sort": "ascending_trailing_20_session_log_return_volatility",
                    },
                    "weak_market_rank1d70_rank10d30_top30": {
                        "production_top_pool": DEFENSIVE_POOL_SIZE,
                        "sort": "descending_0.70_rank_1d_plus_0.30_rank_10d",
                    },
                }[candidate_id],
            },
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": stress,
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout,
            "window_start_metrics": windows,
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
            "weak_market_day_count": int(weak_market.sum()),
        }
        run_cache[candidate_id] = (daily, actions)

    current = results["production_order"]["metrics_0_30pct"]
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
        raise RuntimeError("defensive rerank baseline drifted from checkpoint")

    selected_id = max(results, key=lambda key: renewal.selection_key(results[key]))
    selected_order = {
        "production_order": production_order,
        "weak_market_low_vol_top30": defensive,
        "weak_market_rank1d70_rank10d30_top30": short_term_order,
    }[selected_id]
    selected_daily, selected_actions = run_cache[selected_id]
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        selected_order,
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
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("defensive rerank replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATES),
        "only_change": (
            "on production-defined weak-market days, compare two bounded entry-only "
            "reranks inside the production top-30 pool; exits retain production score"
        ),
        "pit_contract": {
            "signal_date_close_only": True,
            "future_rows_used": False,
            "qfq_common_scale_invariant": True,
        },
        "results": results,
        "walk_forward_selection_evidence": robustness.walk_forward_selections(results),
        "selected_candidate_id": selected_id,
        "selected_reason": (
            "aggregate_2022_2024 training selection; 2025 is confirmation only"
        ),
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
                "results": results,
                "walk_forward": result["walk_forward_selection_evidence"],
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
