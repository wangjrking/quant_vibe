"""Single blind 2026 validation for the frozen 10D daily-zscore candidate."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

import build_expanding_pit_oof_baselines_20260829 as base
from build_10d_daily_zscore_target_dev2022_2024_20260830 import daily_zscore, metrics


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
PARENT = DATA_DIR / "reports" / "model_agent_10d_daily_zscore_target_dev2022_2024_20260830"
DEVELOPMENT = PARENT / "build_r1"
CONFIRMATION = PARENT / "confirmation_2025_r1"
OUT = PARENT / "validation_2026_r1"
CONTRACT = PARENT / "validation_contract_2026.json"
CALENDAR_2025 = base.CALENDAR_2025
CALENDAR_2026 = DATA_DIR / "runtime" / "agent_workspaces" / "data-ingestion-agent" / "work" / "v7_blind_validation_official_trade_cal_20260105_20260813_r1" / "official_trade_cal.duckdb"


def open_dates(path: Path, table: str) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"official calendar missing: {path}")
    connection = duckdb.connect(str(path), read_only=True)
    try:
        return [str(row[0]) for row in connection.execute(f"SELECT cal_date FROM {base.quote(table)} WHERE is_open = 1 ORDER BY cal_date").fetchall()]
    finally:
        connection.close()


def model_predictions(train: pd.DataFrame, test: pd.DataFrame, info: dict, target: pd.Series) -> np.ndarray:
    matrix_train = train[info["resolved"]].astype("float32")
    matrix_test = test[info["resolved"]].astype("float32")
    model = xgb.XGBRegressor(**base.model_params(info["metadata"]))
    model.fit(matrix_train, target.astype("float32"), sample_weight=base.daily_weights(train, info["metadata"].get("sample_weight_config")), verbose=False)
    first = model.predict(matrix_test).astype("float64")
    second = model.predict(matrix_test).astype("float64")
    third = model.predict(matrix_test).astype("float64")
    if not (np.array_equal(first, second) and np.array_equal(first, third)):
        raise RuntimeError("deterministic replay failure")
    return first


def quality(frame: pd.DataFrame, name: str) -> None:
    if (
        frame.duplicated(["trade_date", "stock_code"]).any()
        or frame.stock_code.astype(str).str.endswith(".BJ").any()
        or frame.pred_prob.isna().any()
        or not np.isfinite(frame.pred_prob).all()
    ):
        raise RuntimeError(f"{name}: quality failure")


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing second validation: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    development = json.loads((DEVELOPMENT / "evaluation_summary.json").read_text(encoding="utf-8"))
    confirmation = json.loads((CONFIRMATION / "evaluation_summary.json").read_text(encoding="utf-8"))
    if (
        contract["candidate_id"] != development["candidate_id"]
        or contract["candidate_id"] != confirmation["candidate_id"]
        or development["decision"] != "development_pass_waiting_for_2025_confirmation"
        or confirmation["decision"] != "confirmed_ready_for_audit"
        or not all(development["acceptance_gate"].values())
        or not all(confirmation["acceptance_gate"].values())
    ):
        raise RuntimeError("frozen pre-2026 evidence is not eligible for validation")

    dates_2025 = open_dates(CALENDAR_2025, "official_trade_calendar_binding")
    dates_2026 = open_dates(CALENDAR_2026, "official_trade_cal")
    calendar = dates_2025 + dates_2026
    if calendar != sorted(calendar) or len(calendar) != len(set(calendar)):
        raise RuntimeError("official calendar continuity failure")
    start, end = contract["validation_window"]["start"], contract["validation_window"]["end"]
    test_dates = [date for date in dates_2026 if start <= date <= end]
    if not test_dates or test_dates[0] != start or test_dates[-1] != end:
        raise RuntimeError("validation official-calendar range failure")
    maturity_sessions = 12
    start_index = calendar.index(start)
    if start_index < maturity_sessions:
        raise RuntimeError("insufficient embargo calendar history")
    train_end = calendar[start_index - maturity_sessions]

    _, prepared = base.preflight(base.read_open_dates())
    info = prepared["10d"]
    train_start = info["folds"][0]["train_start"]
    OUT.mkdir(parents=True)
    base.dump_json(
        OUT / "preflight.json",
        {
            "validation_contract_sha256": base.sha256_file(CONTRACT),
            "development_summary_sha256": base.sha256_file(DEVELOPMENT / "evaluation_summary.json"),
            "confirmation_summary_sha256": base.sha256_file(CONFIRMATION / "evaluation_summary.json"),
            "calendar_2025_sha256": base.sha256_file(CALENDAR_2025),
            "calendar_2026_sha256": base.sha256_file(CALENDAR_2026),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": start,
            "test_end": end,
            "official_open_days": len(test_dates),
            "validation_2026_read": True,
            "validation_2026_end_mature_label": end,
            "production_unchanged": True,
            "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
        },
    )

    feature_db = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_db = duckdb.connect(str(base.LABEL_DB), read_only=True)
    try:
        train = base.query_features(feature_db, info["resolved"], train_start, train_end).merge(
            base.query_labels(label_db, "executable_10d_open_return", train_start, train_end),
            on=["trade_date", "stock_code"], how="inner", validate="one_to_one"
        ).dropna(subset=["target"]).reset_index(drop=True)
        test = base.query_features(feature_db, info["resolved"], start, end).merge(
            base.query_labels(label_db, "executable_10d_open_return", start, end),
            on=["trade_date", "stock_code"], how="left", validate="one_to_one"
        )
    finally:
        feature_db.close()
        label_db.close()
    baseline_scores = model_predictions(train, test, info, train["target"])
    candidate_scores = model_predictions(train, test, info, daily_zscore(train))
    baseline = test[["trade_date", "stock_code", "target"]].copy()
    baseline["pred_prob"] = baseline_scores
    candidate = test[["trade_date", "stock_code", "target"]].copy()
    candidate["pred_prob"] = candidate_scores
    quality(baseline, "baseline")
    quality(candidate, "candidate")
    if not baseline[["trade_date", "stock_code"]].equals(candidate[["trade_date", "stock_code"]]):
        raise RuntimeError("same key failure")
    baseline.to_parquet(OUT / "baseline_oof.parquet", index=False)
    candidate.to_parquet(OUT / "candidate_oof.parquet", index=False)
    before = metrics(baseline)
    after = metrics(candidate)
    delta = {key: after[key] - before[key] for key in before}
    gate = {
        "rank_ic_not_weaker": delta["rank_ic_mean"] >= 0,
        "top1_top3_top5_top10_not_weaker": all(delta[f"top{count}_excess_mean"] >= 0 for count in (1, 3, 5, 10)),
        "same_key_finite_no_bj_no_duplicate": True,
        "deterministic_replay_3_of_3": True,
    }
    summary = {
        "candidate_id": contract["candidate_id"],
        "stage": "blind_validation_2026_mature_prefix_only",
        "same_key_rows": len(candidate),
        "baseline": before,
        "candidate": after,
        "delta": delta,
        "acceptance_gate": gate,
        "decision": "validated_research_candidate_ready_for_audit" if all(gate.values()) else "reject_no_further_search",
        "validation_2026_read": True,
        "validation_2026_end_mature_label": end,
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    }
    base.dump_json(OUT / "evaluation_summary.json", summary)
    base.dump_json(
        OUT / "audit_handoff.json",
        {
            "status": summary["decision"],
            "baseline_oof_sha256": base.sha256_file(OUT / "baseline_oof.parquet"),
            "candidate_oof_sha256": base.sha256_file(OUT / "candidate_oof.parquet"),
            "summary_sha256": base.sha256_file(OUT / "evaluation_summary.json"),
            "validation_contract_sha256": base.sha256_file(CONTRACT),
            "production_unchanged": True,
            "allow_next_layer_continue": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
