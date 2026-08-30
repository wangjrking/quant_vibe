from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import duckdb
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
import research_v260_fixed10_weak_market_size_tail_guard_20260822 as scorecard
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


FEATURE_DB = (
    REPO
    / "quant/data_file/experimental_assets/stock_abnormal_announcements_v1/"
    "l2_stock_abnormal_announcement_signal.duckdb"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_abnormal_announcement_proxy_20260822"
)
CASES = {
    "announcement_gate_off": {
        "ordinary_window": 0,
        "severe_window": 0,
        "warning_window": 0,
    },
    "ordinary_1d_only": {
        "ordinary_window": 1,
        "severe_window": 0,
        "warning_window": 0,
    },
    "severe_5d_only": {
        "ordinary_window": 0,
        "severe_window": 5,
        "warning_window": 0,
    },
    "severe_3d_control": {
        "ordinary_window": 0,
        "severe_window": 3,
        "warning_window": 0,
    },
    "severe_10d_control": {
        "ordinary_window": 0,
        "severe_window": 10,
        "warning_window": 0,
    },
    "risk_warning_5d_only": {
        "ordinary_window": 0,
        "severe_window": 0,
        "warning_window": 5,
    },
    "combined_1d_5d_5d": {
        "ordinary_window": 1,
        "severe_window": 5,
        "warning_window": 5,
    },
}


def load_features():
    connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    try:
        return connection.execute(
            """
            SELECT *
            FROM stock_abnormal_announcement_signal
            WHERE signal_date < '20260101'
            ORDER BY signal_date, stock_code
            """
        ).df()
    finally:
        connection.close()


def announcement_block_matrix(
    dates: np.ndarray,
    stocks: np.ndarray,
    features,
    ordinary_window: int,
    severe_window: int,
    warning_window: int,
) -> tuple[np.ndarray, dict]:
    windows = {
        "abnormal": int(ordinary_window),
        "severe": int(severe_window),
        "risk_warning": int(warning_window),
    }
    for window in windows.values():
        if window not in {0, 1, 3, 5, 10}:
            raise ValueError("unsupported announcement window")
    required = {"signal_date", "stock_code"}
    required.update(
        f"{prefix}_count_{window}d"
        for prefix, window in windows.items()
        if window > 0
    )
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"missing announcement fields: {missing}")
    if features.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("announcement feature keys must be unique")

    date_index = {str(value): index for index, value in enumerate(dates)}
    stock_index = {str(value): index for index, value in enumerate(stocks)}
    result = np.zeros((len(dates), len(stocks)), dtype=np.bool_)
    matched_rows = 0
    component_counts = {key: 0 for key in windows}
    columns = ["signal_date", "stock_code"] + [
        f"{prefix}_count_{window}d"
        for prefix, window in windows.items()
        if window > 0
    ]
    for row in features[columns].itertuples(index=False):
        d_idx = date_index.get(str(row.signal_date))
        s_idx = stock_index.get(str(row.stock_code))
        if d_idx is None or s_idx is None:
            continue
        blocked = False
        for prefix, window in windows.items():
            active = window > 0 and int(getattr(row, f"{prefix}_count_{window}d")) > 0
            component_counts[prefix] += int(active)
            blocked = blocked or active
        if blocked:
            result[d_idx, s_idx] = True
            matched_rows += 1
    return result, {
        "windows": windows,
        "component_stock_dates": component_counts,
        "blocked_stock_dates": int(result.sum()),
        "blocked_signal_dates": int(np.any(result, axis=1).sum()),
        "matched_rows": int(matched_rows),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/"
            "pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered announcement proxy development")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    quality_block = quality.maintenance_quality_block(score, True)
    features = load_features()
    results = {}
    run_cache = {}
    masks = {}
    for case_id, case in CASES.items():
        block, block_audit = announcement_block_matrix(
            arrays["dates"], arrays["stocks"], features, **case
        )
        masks[case_id] = block
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=block,
            maintenance_buy_block_mask_override=quality_block | block,
        )
        daily, actions = round1.run_fixed10(
            harness, arrays, protocol, definition, score, order, policy,
            round1.DEVELOPMENT_END, slip=round1.BASELINE_COST,
            record_actions=True, **common,
        )
        stress_daily, stress_actions = round1.run_fixed10(
            harness, arrays, protocol, definition, score, order, policy,
            round1.DEVELOPMENT_END, slip=round1.STRESS_COST,
            record_actions=True, **common,
        )
        metrics = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        )
        results[case_id] = {
            "rule": case,
            "block_audit": block_audit,
            "metrics_0_30pct": metrics,
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily, stress_actions, research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "utility": scorecard.utility(metrics),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
        }
        run_cache[case_id] = (daily, actions)

    baseline_id = "announcement_gate_off"
    baseline = results[baseline_id]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(
            baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("announcement proxy baseline drifted")
    deltas = {
        case_id: float(item["utility"] - baseline["utility"])
        for case_id, item in results.items() if case_id != baseline_id
    }
    raw_winner = max(
        results,
        key=lambda key: (
            all(value > 0 for value in results[key]["metrics_0_30pct"]["annual_returns"].values()),
            results[key]["utility"],
        ),
    )
    winner = results[raw_winner]
    severe_neighbors_positive = bool(
        deltas["severe_3d_control"] > 0
        and deltas["severe_5d_only"] > 0
        and deltas["severe_10d_control"] > 0
    )
    accepted = bool(
        raw_winner == "severe_5d_only"
        and severe_neighbors_positive
        and winner["metrics_0_30pct"]["cagr"] > baseline["metrics_0_30pct"]["cagr"]
        and winner["metrics_0_30pct"]["sharpe"] > baseline["metrics_0_30pct"]["sharpe"]
        and winner["metrics_0_30pct"]["max_drawdown"] <= baseline["metrics_0_30pct"]["max_drawdown"]
        and winner["metrics_0_65pct"]["cumulative_return"] > baseline["metrics_0_65pct"]["cumulative_return"]
        and all(value > 0 for value in winner["metrics_0_30pct"]["annual_returns"].values())
    )
    selected = raw_winner if accepted else baseline_id
    selected_daily, selected_actions = run_cache[selected]
    selected_rule = CASES[selected]
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness, arrays, protocol, definition, score, order, policy,
        round1.DEVELOPMENT_END, slip=round1.BASELINE_COST,
        record_actions=True, simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=masks[selected],
        maintenance_buy_block_mask_override=quality_block | masks[selected],
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions) == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("announcement proxy replay failed")
    result = {
        "status": "proxy_rule_supported_pre2026" if accepted else "proxy_rule_rejected_pre2026",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "only_rule_changed": (
            "block new entries and maintenance top-ups after a PIT-visible abnormal "
            "announcement; never force an existing holding to exit"
        ),
        "source_role": (
            "historical announcement proxy used only to test mechanism; it is not "
            "substituted for the three Tushare runtime interfaces"
        ),
        "candidate_budget": CASES,
        "results": results,
        "utility_deltas_vs_baseline": deltas,
        "raw_winner": raw_winner,
        "severe_window_neighborhood_positive": severe_neighbors_positive,
        "accepted": accepted,
        "selected_candidate": selected,
        "selected_rule": selected_rule,
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
