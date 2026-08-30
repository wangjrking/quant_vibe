from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_lowrisk_score_tuning_20260821 as base
import research_v260_tushare_risk_event_gate_20260822 as gate_v1


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_tushare_risk_event_rules_20260822"
)

BASELINE_COST = 0.003
STRESS_COST = 0.0065

RULES = {
    "event_start_only": {
        "ordinary_shock": "diagnostic_only",
        "severe_shock": "block_new_entry_on_event_day",
        "exchange_alert": "block_new_entry_from_start_through_3_sessions",
        "needs_score_deterioration": False,
    },
    "all_event_momentum_confirmed": {
        "ordinary_shock": "block_new_entry_only_when_10d_rank_below_7d_mean",
        "severe_shock": "block_new_entry_only_when_10d_rank_below_7d_mean",
        "exchange_alert": "block_new_entry_only_when_10d_rank_below_7d_mean",
        "needs_score_deterioration": True,
    },
    "focused_momentum_confirmed": {
        "ordinary_shock": "diagnostic_only",
        "severe_shock": "block_new_entry_on_event_day_when_10d_rank_below_7d_mean",
        "exchange_alert": "block_new_entry_from_start_through_3_sessions_when_10d_rank_below_7d_mean",
        "needs_score_deterioration": True,
    },
}


def _feature_matrix(
    dates: np.ndarray,
    stocks: np.ndarray,
    features: pd.DataFrame,
    column: str,
) -> np.ndarray:
    if column not in features.columns:
        raise ValueError(f"missing Tushare risk-event feature: {column}")
    date_index = {str(value): index for index, value in enumerate(dates)}
    stock_index = {str(value): index for index, value in enumerate(stocks)}
    result = np.zeros((len(dates), len(stocks)), dtype=np.bool_)
    for row in features[["signal_date", "stock_code", column]].itertuples(index=False):
        d_idx = date_index.get(str(row.signal_date))
        s_idx = stock_index.get(str(row.stock_code))
        if d_idx is not None and s_idx is not None and int(getattr(row, column)) > 0:
            result[d_idx, s_idx] = True
    return result


def build_rule_masks(
    arrays: dict[str, np.ndarray],
    features: pd.DataFrame,
    rolling_nanmean,
) -> tuple[dict[str, np.ndarray], dict]:
    dates = arrays["dates"]
    stocks = arrays["stocks"]
    ordinary_1d = _feature_matrix(dates, stocks, features, "shock_count_1d")
    severe_1d = _feature_matrix(dates, stocks, features, "high_shock_count_1d")
    alert_start_3d = _feature_matrix(dates, stocks, features, "alert_start_count_3d")
    alert_active = _feature_matrix(dates, stocks, features, "alert_active_count")

    raw_10d = arrays["rank_10d"].astype(np.float32)
    smooth_10d = rolling_nanmean(raw_10d, 7)
    deteriorating = np.isfinite(raw_10d) & np.isfinite(smooth_10d) & (raw_10d < smooth_10d)

    masks = {
        "event_start_only": severe_1d | alert_start_3d,
        "all_event_momentum_confirmed":
            (ordinary_1d | severe_1d | alert_active) & deteriorating,
        "focused_momentum_confirmed":
            (severe_1d | alert_start_3d) & deteriorating,
    }
    audit = {
        "ordinary_shock_1d_stock_dates": int(ordinary_1d.sum()),
        "severe_shock_1d_stock_dates": int(severe_1d.sum()),
        "alert_start_3d_stock_dates": int(alert_start_3d.sum()),
        "alert_active_stock_dates": int(alert_active.sum()),
        "deteriorating_stock_dates": int(deteriorating.sum()),
        "rule_blocked_stock_dates": {key: int(value.sum()) for key, value in masks.items()},
        "rule_blocked_signal_dates": {
            key: int(np.any(value, axis=1).sum()) for key, value in masks.items()
        },
    }
    return masks, audit


def _metrics(frame: pd.DataFrame, start: str, end: str) -> dict:
    return base.metrics(base.interval(frame, start, end))


def _delta(candidate: dict, baseline: dict) -> dict:
    return {
        key: float(candidate[key] - baseline[key])
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
            "average_positions",
        )
    }


def _monthly_return_deltas(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    start: str,
    end: str,
) -> list[dict]:
    result = []
    left = base.interval(baseline, start, end).copy()
    right = base.interval(candidate, start, end).copy()
    left["month"] = left["date"].astype(str).str[:6]
    right["month"] = right["date"].astype(str).str[:6]
    left_values = left.groupby("month")["return"].apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    right_values = right.groupby("month")["return"].apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    for month in sorted(set(left_values.index) | set(right_values.index)):
        baseline_value = float(left_values.loc[month])
        candidate_value = float(right_values.loc[month])
        result.append(
            {
                "month": str(month),
                "baseline_return": baseline_value,
                "candidate_return": candidate_value,
                "delta": candidate_value - baseline_value,
            }
        )
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    event_manifest, features = gate_v1.load_event_assets()
    harness = base.load_harness()
    protocol, rules, formal_manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    masks, mask_audit = build_rule_masks(
        arrays, features, harness.v174.v94.rolling_nanmean
    )

    event_dates = features.loc[
        (features["shock_count_1d"] > 0)
        | (features["high_shock_count_1d"] > 0)
        | (features["alert_start_count_1d"] > 0),
        "signal_date",
    ].astype(str)
    evaluation_start = str(event_dates.min())
    evaluation_end = base.VALIDATION_END

    baseline_daily, baseline_actions = base.run_shell(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        "production_shell",
        evaluation_end,
        actions=True,
    )
    baseline_metrics = _metrics(baseline_daily, evaluation_start, evaluation_end)
    baseline_stress_daily = gate_v1.run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        np.zeros_like(score, dtype=np.bool_),
        evaluation_end,
        STRESS_COST,
    )
    baseline_stress_metrics = _metrics(
        baseline_stress_daily, evaluation_start, evaluation_end
    )

    rows = []
    artifacts = {}
    for rule_id, blocked in masks.items():
        daily, actions = gate_v1.run_with_entry_gate(
            harness,
            arrays,
            protocol,
            score,
            order,
            definition,
            blocked,
            evaluation_end,
            BASELINE_COST,
            record_actions=True,
        )
        repeat_daily, repeat_actions = gate_v1.run_with_entry_gate(
            harness,
            arrays,
            protocol,
            score,
            order,
            definition,
            blocked,
            evaluation_end,
            BASELINE_COST,
            record_actions=True,
        )
        deterministic = (
            gate_v1.frame_hash(daily) == gate_v1.frame_hash(repeat_daily)
            and gate_v1.frame_hash(actions) == gate_v1.frame_hash(repeat_actions)
        )
        if not deterministic:
            raise RuntimeError(f"deterministic replay failed: {rule_id}")

        stress_daily = gate_v1.run_with_entry_gate(
            harness,
            arrays,
            protocol,
            score,
            order,
            definition,
            blocked,
            evaluation_end,
            STRESS_COST,
        )
        metrics = _metrics(daily, evaluation_start, evaluation_end)
        stress_metrics = _metrics(stress_daily, evaluation_start, evaluation_end)
        delta = _delta(metrics, baseline_metrics)
        action_changes = gate_v1.action_summary(
            baseline_actions, actions, evaluation_start
        )
        monthly_deltas = _monthly_return_deltas(
            baseline_daily, daily, evaluation_start, evaluation_end
        )
        nonzero_months = [item for item in monthly_deltas if abs(item["delta"]) > 1e-12]
        positive_months = [item for item in nonzero_months if item["delta"] > 0.0]
        fragile_small_action_sample = (
            action_changes["baseline_only_buy_keys"] < 20
            or action_changes["candidate_only_buy_keys"] < 20
        )
        pass_all = bool(
            metrics["cumulative_return"] > baseline_metrics["cumulative_return"]
            and metrics["sharpe"] >= baseline_metrics["sharpe"]
            and metrics["max_drawdown"] <= baseline_metrics["max_drawdown"]
            and stress_metrics["cumulative_return"]
            > baseline_stress_metrics["cumulative_return"]
        )
        rows.append(
            {
                "rule_id": rule_id,
                "passed_all_exploratory_gates": pass_all,
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                **{f"candidate_{key}": value for key, value in metrics.items()},
                **{f"delta_{key}": value for key, value in delta.items()},
                "stress_baseline_cumulative_return": baseline_stress_metrics["cumulative_return"],
                "stress_candidate_cumulative_return": stress_metrics["cumulative_return"],
                "positive_delta_months": len(positive_months),
                "nonzero_delta_months": len(nonzero_months),
                "fragile_small_action_sample": fragile_small_action_sample,
                **action_changes,
            }
        )
        daily.to_csv(OUTPUT_ROOT / f"{rule_id}_daily.csv", index=False, encoding="utf-8-sig")
        actions.to_csv(
            OUTPUT_ROOT / f"{rule_id}_actions.csv", index=False, encoding="utf-8-sig"
        )
        artifacts[rule_id] = {
            "definition": RULES[rule_id],
            "metrics_0_30pct": metrics,
            "delta_0_30pct": delta,
            "metrics_0_65pct": stress_metrics,
            "action_changes": action_changes,
            "monthly_deltas": monthly_deltas,
            "fragile_small_action_sample": fragile_small_action_sample,
            "deterministic_replay": deterministic,
            "passed_all_exploratory_gates": pass_all,
        }

    frame = pd.DataFrame(rows).sort_values(
        ["passed_all_exploratory_gates", "candidate_sharpe", "candidate_cumulative_return"],
        ascending=[False, False, False],
    )
    frame.to_csv(OUTPUT_ROOT / "rule_comparison.csv", index=False, encoding="utf-8-sig")
    best_rule = str(frame.iloc[0]["rule_id"])
    any_passed = bool(frame["passed_all_exploratory_gates"].any())
    result = {
        "schema_version": 1,
        "status": (
            "exploratory_rule_found_but_fragile_not_promotable"
            if any_passed
            else "no_rule_beats_production_on_all_gates"
        ),
        "source_strategy": rules["strategy_id"],
        "evaluation_window": [evaluation_start, evaluation_end],
        "data_limit": {
            "event_history_is_mostly_2026": True,
            "independent_validation_available": False,
            "production_promotion_allowed": False,
        },
        "fixed_rule_budget": list(RULES),
        "baseline_0_30pct": baseline_metrics,
        "baseline_0_65pct": baseline_stress_metrics,
        "mask_audit": mask_audit,
        "rules": artifacts,
        "best_exploratory_rule": best_rule,
        "best_exploratory_rule_passed_all_gates": bool(
            artifacts[best_rule]["passed_all_exploratory_gates"]
        ),
        "event_asset_manifest": event_manifest,
        "access": access,
        "formal_manifests": formal_manifests,
        "production_modified": False,
    }
    (OUTPUT_ROOT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "window": result["evaluation_window"],
                "best": best_rule,
                "baseline": baseline_metrics,
                "rules": {
                    key: {
                        "metrics": value["metrics_0_30pct"],
                        "delta": value["delta_0_30pct"],
                        "stress": value["metrics_0_65pct"],
                        "passed": value["passed_all_exploratory_gates"],
                    }
                    for key, value in artifacts.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
