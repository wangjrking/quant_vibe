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
ANNOUNCEMENT_ROOT = REPO / "quant/data_file/experimental_assets/stock_abnormal_announcements_v1"
FEATURE_PATH = ANNOUNCEMENT_ROOT / "l2_stock_abnormal_announcement_signal.duckdb"
MANIFEST_PATH = ANNOUNCEMENT_ROOT / "manifest.json"
OUTPUT_ROOT = REPO / "quant/data_file/reports/strategy_agent_v260_abnormal_announcement_gate_20260822"

sys.path.insert(0, str(MAIN_ROOT))
import research_v260_lowrisk_score_tuning_20260821 as base


WINDOWS = (1, 3, 5, 10)
BASELINE_COST = 0.003
STRESS_COST = 0.0065


def load_announcement_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_l1_l2_ready":
        raise PermissionError("announcement L1/L2 candidate is not ready")
    if manifest["quality"]["l1_duplicate_keys"] or manifest["quality"]["l2_duplicate_keys"]:
        raise PermissionError("announcement L1/L2 duplicate key gate failed")
    if manifest["quality"]["l2_bj_rows"] or manifest["pit_contract"]["same_day_visibility_rows"]:
        raise PermissionError("announcement L2 boundary gate failed")
    return manifest


def load_feature_frame() -> pd.DataFrame:
    connection = duckdb.connect(str(FEATURE_PATH), read_only=True)
    try:
        frame = connection.execute(
            "SELECT * FROM stock_abnormal_announcement_signal ORDER BY signal_date, stock_code"
        ).df()
    finally:
        connection.close()
    if frame.duplicated(["signal_date", "stock_code"]).any():
        raise RuntimeError("announcement feature duplicate key")
    return frame


def feature_matrix(
    dates: np.ndarray,
    stocks: np.ndarray,
    features: pd.DataFrame,
    window: int,
) -> tuple[np.ndarray, dict]:
    column = f"abnormal_count_{int(window)}d"
    if column not in features:
        raise ValueError(f"missing feature column: {column}")
    date_index = {str(value): index for index, value in enumerate(dates)}
    stock_index = {str(value): index for index, value in enumerate(stocks)}
    result = np.zeros((len(dates), len(stocks)), dtype=np.bool_)
    matched = 0
    for row in features[["signal_date", "stock_code", column]].itertuples(index=False):
        d_idx = date_index.get(str(row.signal_date))
        s_idx = stock_index.get(str(row.stock_code))
        if d_idx is None or s_idx is None:
            continue
        if int(getattr(row, column)) > 0:
            result[d_idx, s_idx] = True
            matched += 1
    return result, {
        "window_sessions": int(window),
        "matched_stock_dates": int(result.sum()),
        "matched_feature_rows": matched,
        "blocked_signal_days": int(np.any(result, axis=1).sum()),
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
        raise ValueError("announcement gate shape mismatch")
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


def evaluate_case(daily: pd.DataFrame, name: str, window: int) -> dict:
    evaluation = base.evaluate(daily, {"case": name, "announcement_window_sessions": window})
    return evaluation


def flatten(row: dict) -> dict:
    output = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for block in ("train_2022_2024", "holdout_2025", "pre2026"):
        for key, value in row[block].items():
            if not isinstance(value, dict):
                output[f"{block}_{key}"] = value
    return output


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def interval_metrics(daily: pd.DataFrame, start: str, end: str) -> dict:
    return base.metrics(base.interval(daily, start, end))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    announcement_manifest = load_announcement_manifest()
    features = load_feature_frame()
    harness = base.load_harness()
    protocol, rules, formal_manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)

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
    empty_gate = np.zeros_like(score, dtype=np.bool_)
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
        raise RuntimeError("announcement entry-gate shell does not reproduce production baseline")

    baseline_evaluation = evaluate_case(baseline_daily, "production_baseline", 0)
    matrices: dict[int, np.ndarray] = {}
    matrix_audit: dict[str, dict] = {}
    development_rows = []
    for window in WINDOWS:
        matrix, audit = feature_matrix(arrays["dates"], arrays["stocks"], features, window)
        matrices[window] = matrix
        matrix_audit[str(window)] = audit
        daily, _ = run_with_entry_gate(
            harness,
            arrays,
            protocol,
            score,
            order,
            definition,
            matrix,
            base.PRE2026_END,
            BASELINE_COST,
        )
        development_rows.append(evaluate_case(daily, f"entry_block_{window}d", window))

    winner = max(
        development_rows,
        key=lambda item: (
            item["selection_utility"],
            item["pre2026"]["sharpe"],
            item["pre2026"]["cagr"],
            -item["pre2026"]["max_drawdown"],
            -item["announcement_window_sessions"],
        ),
    )
    development_pass = bool(
        winner["selection_utility"] > baseline_evaluation["selection_utility"]
        and winner["pre2026"]["cagr"] > baseline_evaluation["pre2026"]["cagr"]
        and winner["pre2026"]["sharpe"] >= baseline_evaluation["pre2026"]["sharpe"]
        and winner["pre2026"]["max_drawdown"] <= baseline_evaluation["pre2026"]["max_drawdown"] * 1.10
    )
    winner_window = int(winner["announcement_window_sessions"])
    winner_daily, winner_actions = run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        matrices[winner_window],
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
        matrices[winner_window],
        base.VALIDATION_END,
        BASELINE_COST,
        record_actions=True,
    )
    deterministic = {
        "daily": frame_hash(winner_daily) == frame_hash(repeat_daily),
        "actions": frame_hash(winner_actions) == frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("announcement strategy deterministic replay failed")

    baseline_validation = interval_metrics(baseline_daily, base.VALIDATION_START, base.VALIDATION_END)
    winner_validation = interval_metrics(winner_daily, base.VALIDATION_START, base.VALIDATION_END)
    validation_pass = bool(
        development_pass
        and winner_validation["cumulative_return"] > baseline_validation["cumulative_return"]
        and winner_validation["sharpe"] >= baseline_validation["sharpe"]
        and winner_validation["max_drawdown"] <= baseline_validation["max_drawdown"] * 1.10
    )

    baseline_stress_daily, _ = run_with_entry_gate(
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
    winner_stress_daily, _ = run_with_entry_gate(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        matrices[winner_window],
        base.VALIDATION_END,
        STRESS_COST,
    )
    stress = {
        "cost_each_side": STRESS_COST,
        "pre2026": {
            "baseline": interval_metrics(baseline_stress_daily, base.FIRST_BUY, base.PRE2026_END),
            "candidate": interval_metrics(winner_stress_daily, base.FIRST_BUY, base.PRE2026_END),
        },
        "validation_2026": {
            "baseline": interval_metrics(baseline_stress_daily, base.VALIDATION_START, base.VALIDATION_END),
            "candidate": interval_metrics(winner_stress_daily, base.VALIDATION_START, base.VALIDATION_END),
        },
    }

    pd.DataFrame([flatten(row) for row in development_rows]).sort_values(
        "selection_utility", ascending=False
    ).to_csv(OUTPUT_ROOT / "announcement_window_search_pre2026.csv", index=False, encoding="utf-8-sig")
    baseline_daily.to_csv(OUTPUT_ROOT / "baseline_daily.csv", index=False, encoding="utf-8-sig")
    winner_daily.to_csv(OUTPUT_ROOT / "candidate_daily.csv", index=False, encoding="utf-8-sig")
    baseline_actions.to_csv(OUTPUT_ROOT / "baseline_actions.csv", index=False, encoding="utf-8-sig")
    winner_actions.to_csv(OUTPUT_ROOT / "candidate_actions.csv", index=False, encoding="utf-8-sig")

    result = {
        "schema_version": 1,
        "status": "passed_independent_validation" if validation_pass else "rejected_no_production_change",
        "candidate": "v260_abnormal_announcement_entry_gate_v1",
        "source_strategy": rules["strategy_id"],
        "candidate_semantics": {
            "change": "block new entries with a recent abnormal-volatility announcement",
            "held_position_forced_exit": False,
            "production_score_and_exit_rules_changed": False,
            "selected_window_sessions": winner_window,
        },
        "boundaries": {
            "selection": [base.FIRST_BUY, base.PRE2026_END],
            "independent_validation": [base.VALIDATION_START, base.VALIDATION_END],
            "validation_used_for_selection": False,
        },
        "search_budget": {"windows": list(WINDOWS), "candidate_count": len(WINDOWS)},
        "announcement_asset": announcement_manifest,
        "matrix_audit": matrix_audit,
        "reference_equivalence": reference_equivalence,
        "development": {"baseline": baseline_evaluation, "candidate": winner, "passed": development_pass},
        "validation_2026": {
            "baseline": baseline_validation,
            "candidate": winner_validation,
            "delta": {
                key: float(winner_validation[key] - baseline_validation[key])
                for key in (
                    "cumulative_return",
                    "cagr",
                    "sharpe",
                    "max_drawdown",
                    "turnover_annualized",
                    "average_invested_ratio",
                )
            },
            "passed": validation_pass,
        },
        "stress": stress,
        "deterministic_replay": deterministic,
        "access": access,
        "formal_manifests": formal_manifests,
        "production_modified": False,
    }
    (OUTPUT_ROOT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "selected_window": winner_window}, ensure_ascii=False))
    print(json.dumps(result["validation_2026"], ensure_ascii=False))


if __name__ == "__main__":
    main()
