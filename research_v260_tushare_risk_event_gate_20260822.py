from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
EVENT_ROOT = REPO / "quant/data_file/experimental_assets/tushare_stock_risk_events_v1"
FEATURE_PATH = EVENT_ROOT / "l2_stock_risk_signal.duckdb"
MANIFEST_PATH = EVENT_ROOT / "manifest.json"
OUTPUT_ROOT = REPO / "quant/data_file/reports/strategy_agent_v260_tushare_risk_event_gate_20260822"

BASELINE_COST = 0.003
STRESS_COST = 0.0065

sys.path.insert(0, str(MAIN_ROOT))
import research_v260_lowrisk_score_tuning_20260821 as base


def load_event_assets() -> tuple[dict, pd.DataFrame]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_l1_l2_ready":
        raise PermissionError("Tushare risk-event L1/L2 asset is not ready")
    quality = manifest["quality"]
    if quality["l2_duplicate_event_ids"] or quality["l2_bj_rows"] or quality["feature_duplicate_keys"]:
        raise PermissionError("Tushare risk-event L1/L2 quality gate failed")
    if manifest["pit_contract"]["same_day_visibility_rows"]:
        raise PermissionError("Tushare risk-event PIT gate failed")
    connection = duckdb.connect(str(FEATURE_PATH), read_only=True)
    try:
        features = connection.execute(
            "SELECT * FROM stock_risk_signal ORDER BY signal_date, stock_code"
        ).df()
    finally:
        connection.close()
    if features.duplicated(["signal_date", "stock_code"]).any():
        raise RuntimeError("Tushare risk-event feature duplicate key")
    return manifest, features


def fixed_gate_matrix(
    dates: np.ndarray,
    stocks: np.ndarray,
    features: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    required = ["shock_count_1d", "high_shock_count_5d", "alert_active_count"]
    missing = [column for column in required if column not in features]
    if missing:
        raise ValueError(f"missing fixed Tushare risk-event features: {missing}")
    date_index = {str(value): index for index, value in enumerate(dates)}
    stock_index = {str(value): index for index, value in enumerate(stocks)}
    gate = np.zeros((len(dates), len(stocks)), dtype=np.bool_)
    component_counts = {"ordinary_shock_1d": 0, "severe_shock_5d": 0, "exchange_alert_active": 0}
    matched_rows = 0
    for row in features[["signal_date", "stock_code", *required]].itertuples(index=False):
        d_idx = date_index.get(str(row.signal_date))
        s_idx = stock_index.get(str(row.stock_code))
        if d_idx is None or s_idx is None:
            continue
        ordinary = int(row.shock_count_1d) > 0
        severe = int(row.high_shock_count_5d) > 0
        alert = int(row.alert_active_count) > 0
        if ordinary:
            component_counts["ordinary_shock_1d"] += 1
        if severe:
            component_counts["severe_shock_5d"] += 1
        if alert:
            component_counts["exchange_alert_active"] += 1
        if ordinary or severe or alert:
            gate[d_idx, s_idx] = True
            matched_rows += 1
    return gate, {
        "fixed_rule": {
            "stk_shock": "block new entry for 1 official trading session",
            "stk_high_shock": "block new entry for 5 official trading sessions",
            "stk_alert": "block new entry while official alert interval remains active",
        },
        "component_stock_dates": component_counts,
        "matched_feature_rows": int(matched_rows),
        "blocked_stock_dates": int(gate.sum()),
        "blocked_signal_dates": int(np.any(gate, axis=1).sum()),
    }


def run_with_entry_gate(
    harness,
    arrays,
    protocol,
    score,
    order,
    definition,
    blocked: np.ndarray,
    end_date: str,
    slip: float,
    record_actions: bool = False,
):
    if blocked.shape != score.shape:
        raise ValueError("risk-event gate shape mismatch")
    case_protocol = copy.deepcopy(protocol)
    case_protocol["execution"]["fixed_slippage_ratio"] = float(slip)
    production_selection = harness.v174.selection_mask(arrays, definition["max_rank_deterioration"])
    selection = np.asarray(production_selection, dtype=np.bool_) & ~np.asarray(blocked, dtype=np.bool_)
    multiplier = harness.v260.v258.v252.warmup_multiplier(
        arrays, score, definition, case_protocol, base.FIRST_BUY
    )
    positions = harness.v260.position_schedule(arrays, definition, base.FIRST_BUY)
    return harness.v162.run_case(
        arrays,
        score,
        order,
        definition,
        case_protocol,
        end_date,
        base.FIRST_BUY,
        record_actions=record_actions,
        selection_mask_override=selection,
        candidate_target_multiplier_override=multiplier,
        max_positions_schedule_override=positions,
    )


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def interval_metrics(frame: pd.DataFrame, start_date: str, end_date: str) -> dict:
    return base.metrics(base.interval(frame, start_date, end_date))


def action_summary(baseline: pd.DataFrame, candidate: pd.DataFrame, start_date: str) -> dict:
    left = baseline[baseline["signal_date"].astype(str) >= start_date].copy()
    right = candidate[candidate["signal_date"].astype(str) >= start_date].copy()
    left_buys = set(map(tuple, left[left["action"] == "BUY"][["signal_date", "stock_code"]].astype(str).to_numpy()))
    right_buys = set(map(tuple, right[right["action"] == "BUY"][["signal_date", "stock_code"]].astype(str).to_numpy()))
    return {
        "baseline_buy_actions": int(len(left_buys)),
        "candidate_buy_actions": int(len(right_buys)),
        "baseline_only_buy_keys": int(len(left_buys - right_buys)),
        "candidate_only_buy_keys": int(len(right_buys - left_buys)),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    event_manifest, features = load_event_assets()
    harness = base.load_harness()
    protocol, rules, formal_manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    gate, gate_audit = fixed_gate_matrix(arrays["dates"], arrays["stocks"], features)
    empty_gate = np.zeros_like(gate, dtype=np.bool_)

    baseline_daily, baseline_actions = base.run_shell(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        "production_shell",
        base.VALIDATION_END,
        actions=True,
    )
    reference_daily, reference_actions = run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        empty_gate,
        base.VALIDATION_END,
        BASELINE_COST,
        record_actions=True,
    )
    reference_equivalence = {
        "daily": frame_hash(baseline_daily) == frame_hash(reference_daily),
        "actions": frame_hash(baseline_actions) == frame_hash(reference_actions),
    }
    if not all(reference_equivalence.values()):
        raise RuntimeError("risk-event entry-gate shell does not reproduce production baseline")

    candidate_daily, candidate_actions = run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        gate,
        base.VALIDATION_END,
        BASELINE_COST,
        record_actions=True,
    )
    repeat_daily, repeat_actions = run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        gate,
        base.VALIDATION_END,
        BASELINE_COST,
        record_actions=True,
    )
    deterministic = {
        "daily": frame_hash(candidate_daily) == frame_hash(repeat_daily),
        "actions": frame_hash(candidate_actions) == frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("risk-event candidate deterministic replay failed")

    first_dates = [
        value["min_date"]
        for value in event_manifest["quality"]["apis"].values()
        if value["min_date"] is not None
    ]
    evaluation_start = str(max(first_dates))
    evaluation_start = str(
        features.loc[
            (features["signal_date"].astype(str) > evaluation_start)
            & (
                (features["shock_count_1d"] > 0)
                | (features["high_shock_count_5d"] > 0)
                | (features["alert_active_count"] > 0)
            ),
            "signal_date",
        ].min()
    )
    evaluation_end = base.VALIDATION_END
    baseline_metrics = interval_metrics(baseline_daily, evaluation_start, evaluation_end)
    candidate_metrics = interval_metrics(candidate_daily, evaluation_start, evaluation_end)

    baseline_stress = run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        empty_gate,
        base.VALIDATION_END,
        STRESS_COST,
    )
    candidate_stress = run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        gate,
        base.VALIDATION_END,
        STRESS_COST,
    )
    baseline_stress_metrics = interval_metrics(baseline_stress, evaluation_start, evaluation_end)
    candidate_stress_metrics = interval_metrics(candidate_stress, evaluation_start, evaluation_end)
    passed = bool(
        candidate_metrics["cumulative_return"] > baseline_metrics["cumulative_return"]
        and candidate_metrics["sharpe"] >= baseline_metrics["sharpe"]
        and candidate_metrics["max_drawdown"] <= baseline_metrics["max_drawdown"] * 1.10
        and candidate_stress_metrics["cumulative_return"] > baseline_stress_metrics["cumulative_return"]
    )

    baseline_daily.to_csv(OUTPUT_ROOT / "baseline_daily.csv", index=False, encoding="utf-8-sig")
    candidate_daily.to_csv(OUTPUT_ROOT / "candidate_daily.csv", index=False, encoding="utf-8-sig")
    baseline_actions.to_csv(OUTPUT_ROOT / "baseline_actions.csv", index=False, encoding="utf-8-sig")
    candidate_actions.to_csv(OUTPUT_ROOT / "candidate_actions.csv", index=False, encoding="utf-8-sig")
    result = {
        "schema_version": 1,
        "status": "passed_exploratory_2026" if passed else "rejected_no_production_change",
        "candidate": "v260_tushare_risk_event_entry_gate_v1",
        "source_strategy": rules["strategy_id"],
        "candidate_semantics": {
            "only_change": "new-entry eligibility uses the fixed Tushare risk-event gate",
            "held_position_forced_exit": False,
            "score_exit_portfolio_rules_changed": False,
            **gate_audit["fixed_rule"],
        },
        "data_limit": {
            "pre2026_tuning_performed": False,
            "reason": "the three Tushare interfaces contain only one pre-2026 row in total, so no defensible parameter tuning is possible",
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            "interpretation": "fixed-rule 2026 diagnostic, not a tuned independent promotion test",
        },
        "gate_audit": gate_audit,
        "reference_equivalence": reference_equivalence,
        "deterministic_replay": deterministic,
        "baseline_0_30pct": baseline_metrics,
        "candidate_0_30pct": candidate_metrics,
        "delta_0_30pct": {
            key: float(candidate_metrics[key] - baseline_metrics[key])
            for key in (
                "cumulative_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "turnover_annualized",
                "average_invested_ratio",
            )
        },
        "stress_0_65pct": {
            "baseline": baseline_stress_metrics,
            "candidate": candidate_stress_metrics,
        },
        "action_changes": action_summary(baseline_actions, candidate_actions, evaluation_start),
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
                "window": [evaluation_start, evaluation_end],
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta": result["delta_0_30pct"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
