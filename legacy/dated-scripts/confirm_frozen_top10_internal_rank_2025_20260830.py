"""One-time 2025 confirmation for frozen 1D/10D Top10-internal candidates.

This is not candidate selection.  It replays the already frozen HistGradient
model specification once against the sealed 2025 PIT fold, then applies the
zero-parameter Top10-internal projection.  The query bounds make 2026
unreadable to the build.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics, dump_json
from build_expanding_pit_oof_baselines_20260829 import (
    FEATURE_DB, FEATURE_TABLE, HORIZONS, LABEL_DB, LABEL_TABLE, POLICY_PATH,
    canonical_frame_hash, daily_weights, evaluate, query_features, query_labels,
    quote, sha256_file,
)
from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases


ROOT = Path("quant/data_file/reports/model_agent_frozen_top10_internal_rank_2025_confirmation_20260830")
BASELINE_ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
CONTRACT = ROOT / "confirmation_contract.json"
SPECS = {
    "1d": {
        "candidate_id": "v261_1d_hist_top10_internal_rank_v1",
        "hgb_contract": Path("quant/data_file/reports/model_agent_1d_hist_gradient_structure_20260830/training_contract.json"),
        "mature_end": "20251229",
        "feature_count": 78,
    },
    "10d": {
        "candidate_id": "v261_10d_hist_top10_internal_rank_v1",
        "hgb_contract": Path("quant/data_file/reports/model_agent_10d_hist_gradient_structure_20260830/training_contract.json"),
        "mature_end": "20251215",
        "feature_count": 40,
    },
}
PREFIX = 10


def project(source: pd.DataFrame) -> pd.DataFrame:
    result: list[pd.DataFrame] = []
    for _, group in source.groupby("trade_date", sort=True):
        baseline = group.sort_values(["baseline_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort").copy()
        prefix = baseline.head(PREFIX).sort_values(["candidate_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        final = pd.concat([prefix, baseline.iloc[PREFIX:]], ignore_index=True)
        if set(final.head(PREFIX)["stock_code"]) != set(baseline.head(PREFIX)["stock_code"]):
            raise RuntimeError("blocked_top10_membership_changed")
        final["baseline_rank"] = final["stock_code"].map({code: rank + 1 for rank, code in enumerate(baseline["stock_code"])}).astype("int32")
        final["final_rank"] = np.arange(1, len(final) + 1, dtype="int32")
        final["candidate_raw_score"] = (len(final) - final["final_rank"]).astype("float64")
        result.append(final)
    return pd.concat(result, ignore_index=True).sort_values(["trade_date", "stock_code"], kind="mergesort").reset_index(drop=True)


def metric_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.rename(columns={column: "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]]


def run(horizon_key: str, output: Path) -> int:
    if output.exists():
        raise RuntimeError("blocked_existing_output")
    if horizon_key not in SPECS:
        raise RuntimeError("blocked_horizon")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    spec = SPECS[horizon_key]
    definition = contract["confirmations"][horizon_key]
    if definition["candidate_id"] != spec["candidate_id"] or definition["mature_test_end"] != spec["mature_end"]:
        raise RuntimeError("blocked_confirmation_contract")
    if json.loads(POLICY_PATH.read_text(encoding="utf-8"))["training_window_hard_constraints"]["mode"] != "expanding_available_history":
        raise RuntimeError("blocked_training_window_policy")
    horizon = next(item for item in HORIZONS if item.key == horizon_key)
    metadata = json.loads(horizon.metadata_path.read_text(encoding="utf-8"))
    fconn = duckdb.connect(str(FEATURE_DB), read_only=True)
    lconn = duckdb.connect(str(LABEL_DB), read_only=True)
    try:
        feature_min = str(fconn.execute(f"SELECT min(trade_date) FROM {quote(FEATURE_TABLE)} WHERE trade_date<=?", [definition["train_end"]]).fetchone()[0])
        label_min = str(lconn.execute(f"SELECT min(trade_date) FROM {quote(LABEL_TABLE)} WHERE trade_date<=?", [definition["train_end"]]).fetchone()[0])
        available = {str(row[1]) for row in fconn.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}
        requested = [str(value) for value in metadata["feature_columns"]]
        aliases = resolve_savedmodel_feature_aliases(requested, available)
        if aliases["missing"]:
            raise RuntimeError(f"blocked_feature_aliases:{aliases['missing']}")
        alias_map = dict(aliases["alias_pairs"])
        features = [alias_map.get(name, name) for name in requested]
        if len(features) != spec["feature_count"] or len(features) != len(set(features)):
            raise RuntimeError("blocked_feature_contract")
        train_start = max(feature_min, label_min)
        train_features = query_features(fconn, features, train_start, definition["train_end"])
        train_labels = query_labels(lconn, horizon.label, train_start, definition["train_end"])
        train = train_features.merge(train_labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"])
        test_features = query_features(fconn, features, definition["test_start"], definition["mature_test_end"])
        test_labels = query_labels(lconn, horizon.label, definition["test_start"], definition["mature_test_end"])
        test = test_features.merge(test_labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"])
    finally:
        fconn.close()
        lconn.close()
    if train.empty or test.empty or test["trade_date"].astype(str).str[:4].ne("2025").any():
        raise RuntimeError("blocked_pit_date_window")
    weights = daily_weights(train[["trade_date", "stock_code", "target"]], metadata.get("sample_weight_config"))
    model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=8.0, early_stopping=False, random_state=42)
    matrix_train = train[features].astype("float32")
    matrix_test = test[features].astype("float32")
    model.fit(matrix_train, train["target"].astype("float32"), sample_weight=weights)
    predictions = [model.predict(matrix_test).astype("float64") for _ in range(3)]
    if not (np.array_equal(predictions[0], predictions[1]) and np.array_equal(predictions[0], predictions[2])):
        raise RuntimeError("blocked_hgb_determinism")
    candidate = test[["trade_date", "stock_code", "target"]].copy()
    candidate["candidate_pred_prob"] = predictions[0]
    con = duckdb.connect()
    try:
        baseline = con.execute(
            f"SELECT trade_date, stock_code, pred_prob AS baseline_pred_prob FROM read_parquet('{(BASELINE_ROOT / (horizon_key + '_oof.parquet')).as_posix()}') "
            "WHERE fold_id='fold2025' AND label_mature_within_dev AND trade_date BETWEEN ? AND ? ORDER BY trade_date, stock_code",
            [definition["test_start"], definition["mature_test_end"]],
        ).fetchdf()
    finally:
        con.close()
    same_key = candidate.merge(baseline, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    if len(same_key) != len(candidate) or same_key.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_same_key")
    if same_key["stock_code"].astype(str).str.endswith(".BJ").any() or not np.isfinite(same_key[["target", "baseline_pred_prob", "candidate_pred_prob"]].to_numpy(dtype="float64")).all():
        raise RuntimeError("blocked_quality")
    replays = [project(same_key) for _ in range(3)]
    hashes = [canonical_frame_hash(item, ["trade_date", "stock_code", "candidate_raw_score"]) for item in replays]
    if len(set(hashes)) != 1:
        raise RuntimeError("blocked_projection_determinism")
    oof = replays[0]
    base_metrics, _ = evaluate(metric_frame(oof, "baseline_pred_prob"))
    candidate_metrics, _ = evaluate(metric_frame(oof, "candidate_raw_score"))
    failures = compare_metrics(base_metrics, candidate_metrics)
    changed = float((oof["baseline_rank"] != oof["final_rank"]).mean())
    if changed == 0:
        failures.append("nonzero_effective_change")
    failures = sorted(set(failures))
    output.mkdir(parents=True)
    oof.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
    dump_json(output / "preflight.json", {
        "confirmation_type": "one_time_sealed_2025_confirmation_not_selection",
        "candidate_id": spec["candidate_id"], "horizon": horizon_key,
        "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
        "read_windows": {"train": [train_start, definition["train_end"]], "test": [definition["test_start"], definition["mature_test_end"]], "2026_plus": "not_read"},
        "feature_db_sha256": sha256_file(FEATURE_DB), "label_db_sha256": sha256_file(LABEL_DB),
        "baseline_oof_sha256": sha256_file(BASELINE_ROOT / f"{horizon_key}_oof.parquet"),
        "production_metadata_sha256": sha256_file(horizon.metadata_path), "hgb_contract_sha256": sha256_file(spec["hgb_contract"]),
        "feature_alias_pairs": aliases["alias_pairs"], "features": features,
    })
    summary = {
        "candidate_id": spec["candidate_id"], "horizon": horizon_key,
        "status": "completed_waiting_for_fixed_readonly_audit" if not failures else "reject_no_further_search",
        "confirmation_type": "one_time_sealed_2025_confirmation_not_selection",
        "selection_closed_before_2025": True, "validation_2026_closed": True,
        "same_key_rows": int(len(oof)), "same_key_duplicate_groups": int(oof.duplicated(["trade_date", "stock_code"]).sum()),
        "bj_rows": int(oof["stock_code"].astype(str).str.endswith(".BJ").sum()),
        "null_or_nonfinite_rows": int((~np.isfinite(oof[["target", "baseline_pred_prob", "candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
        "projection": {"mutable_prefix_size": PREFIX, "changed_rank_row_share": changed, "top10_membership_identical": True},
        "baseline_metrics": base_metrics, "candidate_metrics": candidate_metrics, "failed_gates": failures,
        "deterministic": {"passed": True, "replay_candidate_score_sha256": hashes},
        "baseline_score_sha256": canonical_frame_hash(metric_frame(oof, "baseline_pred_prob"), ["trade_date", "stock_code", "pred_prob"]),
        "candidate_score_sha256": canonical_frame_hash(metric_frame(oof, "candidate_raw_score"), ["trade_date", "stock_code", "pred_prob"]),
        "production_unchanged": True, "allow_next_layer_continue": False,
    }
    dump_json(output / "evaluation_summary.json", summary)
    dump_json(output / "hash_inventory.json", {"script_sha256": sha256_file(Path(__file__)), "confirmation_contract_sha256": sha256_file(CONTRACT), "preflight_sha256": sha256_file(output / "preflight.json"), "oof_sha256": sha256_file(output / "baseline_candidate_same_key_oof.parquet"), "summary_sha256": sha256_file(output / "evaluation_summary.json")})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", choices=sorted(SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.horizon, args.output))
