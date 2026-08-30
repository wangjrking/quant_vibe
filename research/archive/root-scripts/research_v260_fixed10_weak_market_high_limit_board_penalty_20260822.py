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
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_size_tail_guard_20260822 as scorecard
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_weak_market_high_limit_board_penalty_20260822"
)
CASES = {
    "production_board_penalty_off": {"penalty": 0.0, "selectable": False},
    "control_high_limit_board_penalty_0025": {"penalty": 0.025, "selectable": False},
    "candidate_high_limit_board_penalty_0050": {"penalty": 0.050, "selectable": True},
    "control_high_limit_board_penalty_0075": {"penalty": 0.075, "selectable": False},
}


def weak_market_board_penalty_order(
    production_score: np.ndarray,
    production_order: np.ndarray,
    high_limit_board: np.ndarray,
    weak_market: np.ndarray,
    penalty: float,
) -> np.ndarray:
    score = np.asarray(production_score, dtype=np.float64)
    original = np.asarray(production_order)
    high_board = np.asarray(high_limit_board, dtype=np.bool_)
    weak = np.asarray(weak_market, dtype=np.bool_)
    if score.shape != original.shape or high_board.shape != (score.shape[1],):
        raise ValueError("board penalty inputs do not align")
    if weak.shape != (score.shape[0],):
        raise ValueError("weak-market vector does not align")
    amount = float(penalty)
    if amount < 0:
        raise ValueError("board penalty cannot be negative")
    result = original.copy()
    if amount == 0:
        return result
    tie_breaker = np.arange(score.shape[1], dtype=np.int64)
    for day in np.flatnonzero(weak):
        row = original[day]
        row_score = score[day, row]
        adjusted = row_score - amount * high_board[row].astype(np.float64)
        safe = np.where(np.isfinite(row_score), adjusted, -np.inf)
        rearranged = np.lexsort((tie_breaker, -safe))
        result[day] = row[rearranged]
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
        raise PermissionError("2026 data entered board-penalty development")
    definition = harness.production_definition(protocol)
    score, production_order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    weak_market = ~regime.strong_market_mask(score, protocol)
    high_board = runtime.core.board_limit_rate(arrays["stocks"].astype(str)) > 0.15
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    maintenance_block = quality.maintenance_quality_block(score, True)
    results = {}
    run_cache = {}
    order_cache = {}
    for case_id, case in CASES.items():
        current_order = weak_market_board_penalty_order(
            score,
            production_order,
            high_board,
            weak_market,
            case["penalty"],
        )
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=np.zeros(score.shape, dtype=np.bool_),
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
        results[case_id] = {
            "selectable": bool(case["selectable"]),
            "weak_market_high_limit_board_score_penalty": float(case["penalty"]),
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "utility": scorecard.utility(metrics),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
            "weak_market_days": int(weak_market.sum()),
            "high_limit_board_stock_count": int(high_board.sum()),
            "order_changed_cells": int((current_order != production_order).sum()),
        }
        run_cache[case_id] = (daily, actions)
        order_cache[case_id] = current_order

    baseline_id = "production_board_penalty_off"
    candidate_id = "candidate_high_limit_board_penalty_0050"
    baseline = results[baseline_id]
    candidate = results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
            )
        )
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
        raise RuntimeError("board-penalty baseline drifted")
    utility_deltas = {
        case_id: float(item["utility"] - baseline["utility"])
        for case_id, item in results.items()
        if case_id != baseline_id
    }
    local_robust = bool(
        utility_deltas["control_high_limit_board_penalty_0025"] > 0
        and utility_deltas["control_high_limit_board_penalty_0075"] > 0
    )
    candidate_passed = bool(
        local_robust
        and candidate["metrics_0_30pct"]["cagr"]
        > baseline["metrics_0_30pct"]["cagr"]
        and candidate["metrics_0_30pct"]["sharpe"]
        > baseline["metrics_0_30pct"]["sharpe"]
        and candidate["metrics_0_30pct"]["max_drawdown"]
        < baseline["metrics_0_30pct"]["max_drawdown"]
        and candidate["metrics_0_65pct"]["cumulative_return"]
        > baseline["metrics_0_65pct"]["cumulative_return"]
        and all(value > 0 for value in candidate["metrics_0_30pct"]["annual_returns"].values())
    )
    selected = candidate_id if candidate_passed else baseline_id
    selected_daily, selected_actions = run_cache[selected]
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order_cache[selected],
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=np.zeros(score.shape, dtype=np.bool_),
        maintenance_buy_block_mask_override=maintenance_block,
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("board-penalty replay failed")
    selected_policy = copy.deepcopy(policy)
    selected_policy["weak_market_high_limit_board_score_penalty"] = CASES[
        selected
    ]["penalty"]
    result = {
        "status": "candidate_confirmed_pre2026_2026_not_opened" if candidate_passed else "candidate_rejected_pre2026_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "formula": "entry_score = production_score - penalty * high_limit_board, weak_market_only",
        "selectable_candidate": candidate_id,
        "nonselectable_controls": [
            "control_high_limit_board_penalty_0025",
            "control_high_limit_board_penalty_0075",
        ],
        "results": results,
        "utility_deltas_vs_baseline": utility_deltas,
        "local_neighborhood_robust": local_robust,
        "candidate_passed": candidate_passed,
        "selected_candidate": selected,
        "selected_policy": selected_policy,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
