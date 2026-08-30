"""One frozen strict-PIT 1D HistGradient research-only candidate build."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_expanding_pit_oof_baselines_20260829 import (
    CALENDAR_2021_2024, DEVELOPMENT_END, FEATURE_DB, FEATURE_TABLE, HORIZONS,
    LABEL_DB, LABEL_TABLE, POLICY_PATH, canonical_frame_hash, daily_weights,
    evaluate, query_features, query_labels, quote, sha256_file,
)
from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics, dump_json
from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases


CANDIDATE_ID = "v261_1d_hist_gradient_regressor_v1"
CONTRACT = Path("quant/data_file/reports/model_agent_1d_hist_gradient_structure_20260830/training_contract.json")
BASELINE_ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
DEFAULT_OUTPUT = Path("quant/data_file/reports/model_agent_1d_hist_gradient_structure_20260830/build_r1")
START = "20100104"
END = "20241231"


def open_dates() -> list[str]:
    connection = duckdb.connect(str(CALENDAR_2021_2024), read_only=True)
    try:
        values = [str(row[0]) for row in connection.execute(
            "SELECT cal_date FROM official_trade_cal WHERE is_open=1 AND cal_date<=? ORDER BY cal_date", [END]
        ).fetchall()]
    finally:
        connection.close()
    if not values or values[-1] != END or values != sorted(set(values)):
        raise RuntimeError("blocked_official_calendar")
    return values


def folds(dates: list[str], train_start: str, maturity_sessions: int) -> list[dict[str, str]]:
    result = []
    for year in ("2022", "2023", "2024"):
        dates_year = [date for date in dates if date.startswith(year)]
        test_start, final_date = dates_year[0], dates_year[-1]
        start_index = dates.index(test_start)
        mature_end = dates[dates.index(final_date) - maturity_sessions]
        result.append({"fold_id": f"fold{year}", "train_start": train_start, "train_end": dates[start_index - maturity_sessions], "test_start": test_start, "test_end": mature_end, "embargo_rule": f"{maturity_sessions} official open sessions"})
    return result


def read_baseline(fold: dict[str, str]) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT trade_date, stock_code, pred_prob AS baseline_pred_prob FROM read_parquet('{BASELINE_ROOT.as_posix()}/1d_oof.parquet') WHERE fold_id=? AND label_mature_within_dev AND trade_date BETWEEN ? AND ? ORDER BY trade_date, stock_code",
            [fold["fold_id"], fold["test_start"], fold["test_end"]],
        ).fetchdf()
    finally:
        con.close()


def run(output: Path) -> int:
    if output.exists():
        raise RuntimeError("blocked_existing_output")
    output.mkdir(parents=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("blocked_contract_mismatch")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy["training_window_hard_constraints"]["mode"] != "expanding_available_history":
        raise RuntimeError("blocked_training_window_policy")
    horizon = next(item for item in HORIZONS if item.key == "1d")
    metadata = json.loads(horizon.metadata_path.read_text(encoding="utf-8"))
    dates = open_dates()
    fconn = duckdb.connect(str(FEATURE_DB), read_only=True)
    lconn = duckdb.connect(str(LABEL_DB), read_only=True)
    try:
        feature_min = str(fconn.execute(f"SELECT min(trade_date) FROM {quote(FEATURE_TABLE)} WHERE trade_date<=?", [END]).fetchone()[0])
        label_min = str(lconn.execute(f"SELECT min(trade_date) FROM {quote(LABEL_TABLE)} WHERE trade_date<=?", [END]).fetchone()[0])
        train_start = max(feature_min, label_min)
        available = {str(row[1]) for row in fconn.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}
        requested = [str(value) for value in metadata["feature_columns"]]
        aliases = resolve_savedmodel_feature_aliases(requested, available)
        if aliases["missing"]:
            raise RuntimeError(f"blocked_feature_aliases:{aliases['missing']}")
        alias_map = dict(aliases["alias_pairs"])
        features = [alias_map.get(value, value) for value in requested]
        if len(features) != 78 or len(set(features)) != len(features):
            raise RuntimeError("blocked_feature_contract")
        frozen_folds = folds(dates, train_start, horizon.maturity_sessions)
        dump_json(output / "preflight.json", {"candidate_id": CANDIDATE_ID, "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"}, "development_window": ["20220101", "20241231"], "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"}, "feature_db_sha256": sha256_file(FEATURE_DB), "label_db_sha256": sha256_file(LABEL_DB), "calendar_sha256": sha256_file(CALENDAR_2021_2024), "production_metadata_sha256": sha256_file(horizon.metadata_path), "features": features, "feature_alias_pairs": aliases["alias_pairs"], "folds": frozen_folds, "production_unchanged": True})
        rows: list[pd.DataFrame] = []
        fold_results: dict[str, object] = {}
        for fold in frozen_folds:
            train_f = query_features(fconn, features, fold["train_start"], fold["train_end"])
            train_y = query_labels(lconn, horizon.label, fold["train_start"], fold["train_end"])
            train = train_f.merge(train_y, on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"])
            weights = daily_weights(train[["trade_date", "stock_code", "target"]], metadata.get("sample_weight_config"))
            test_f = query_features(fconn, features, fold["test_start"], fold["test_end"])
            test_y = query_labels(lconn, horizon.label, fold["test_start"], fold["test_end"])
            test = test_f.merge(test_y, on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"])
            model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=8.0, early_stopping=False, random_state=42)
            matrix_train = train[features].astype("float32"); matrix_test = test[features].astype("float32")
            model.fit(matrix_train, train["target"].astype("float32"), sample_weight=weights)
            predictions = [model.predict(matrix_test).astype("float64") for _ in range(3)]
            if not (np.array_equal(predictions[0], predictions[1]) and np.array_equal(predictions[0], predictions[2])):
                raise RuntimeError(f"blocked_deterministic_prediction:{fold['fold_id']}")
            candidate = test[["trade_date", "stock_code", "target"]].copy(); candidate["candidate_pred_prob"] = predictions[0]
            baseline = read_baseline(fold)
            emitted = candidate.merge(baseline, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            if len(emitted) != len(candidate) or emitted.duplicated(["trade_date", "stock_code"]).any():
                raise RuntimeError(f"blocked_same_key:{fold['fold_id']}")
            if emitted.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(emitted[["target", "candidate_pred_prob", "baseline_pred_prob"]].to_numpy(dtype="float64")).all():
                raise RuntimeError(f"blocked_quality:{fold['fold_id']}")
            base_metrics, _ = evaluate(emitted.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
            candidate_metrics, _ = evaluate(emitted.rename(columns={"candidate_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
            fold_results[fold["fold_id"]] = {"fold": fold, "train_rows": int(len(train)), "test_rows": int(len(emitted)), "baseline_metrics": base_metrics, "candidate_metrics": candidate_metrics, "failed_gates": compare_metrics(base_metrics, candidate_metrics), "candidate_prediction_sha256": canonical_frame_hash(emitted.rename(columns={"candidate_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"])}
            emitted["fold_id"] = fold["fold_id"]; rows.append(emitted)
    finally:
        fconn.close(); lconn.close()
    oof = pd.concat(rows, ignore_index=True)
    base_metrics, _ = evaluate(oof.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
    candidate_metrics, _ = evaluate(oof.rename(columns={"candidate_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
    failures = compare_metrics(base_metrics, candidate_metrics) + [failure for item in fold_results.values() for failure in item["failed_gates"]]
    failures = sorted(set(failures)); passed = not failures
    oof.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
    summary = {"candidate_id": CANDIDATE_ID, "status": "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search", "development_window": ["20220101", "20241231"], "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"}, "same_key_rows": int(len(oof)), "same_key_duplicate_groups": int(oof.duplicated(["trade_date", "stock_code"]).sum()), "bj_rows": int(oof.stock_code.astype(str).str.endswith(".BJ").sum()), "null_or_nonfinite_rows": int((~np.isfinite(oof[["target", "baseline_pred_prob", "candidate_pred_prob"]].to_numpy(dtype="float64")).all(axis=1)).sum()), "aggregate": {"baseline_metrics": base_metrics, "candidate_metrics": candidate_metrics, "failed_gates": compare_metrics(base_metrics, candidate_metrics)}, "folds": fold_results, "hard_gate_passed": passed, "failed_gates": failures, "baseline_score_sha256": canonical_frame_hash(oof.rename(columns={"baseline_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"]), "candidate_score_sha256": canonical_frame_hash(oof.rename(columns={"candidate_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"]), "production_unchanged": True, "allow_next_layer_continue": False}
    dump_json(output / "evaluation_summary.json", summary)
    dump_json(output / "research_candidate_manifest.json", {"candidate_id": CANDIDATE_ID, "approval_status": "research_only_not_for_l5", "decision": summary["status"], "production_unchanged": True, "allow_next_layer_continue": False})
    dump_json(output / "hash_inventory.json", {"script_sha256": sha256_file(Path(__file__)), "contract_sha256": sha256_file(CONTRACT), "preflight_sha256": sha256_file(output / "preflight.json"), "oof_sha256": sha256_file(output / "baseline_candidate_same_key_oof.parquet"), "summary_sha256": sha256_file(output / "evaluation_summary.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(run(DEFAULT_OUTPUT))
