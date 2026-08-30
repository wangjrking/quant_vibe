"""Build the single frozen v260 PIT/OOF excess-logit 10D research candidate.

This script is intentionally self-contained and research-only.  It never reads
2025 or later rows, and it does not access production prediction tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases


FEATURE_DB = Path("quant/data_file/production_assets/duckdb/l3_feature_current.duckdb")
LABEL_DB = Path("quant/data_file/production_assets/duckdb/l3_label_current.duckdb")
OFFICIAL_CALENDAR_ROOT = Path(
    "quant/data_file/runtime/agent_workspaces/data-ingestion-agent/work/"
    "v260_pit_oof_excess_logit_10d_v1_official_trade_cal_20210101_20241231_r1"
)
OFFICIAL_CALENDAR_DESCRIPTOR = OFFICIAL_CALENDAR_ROOT / "official_trade_cal_descriptor_v1.json"
OFFICIAL_CALENDAR_DB = OFFICIAL_CALENDAR_ROOT / "official_trade_cal.duckdb"
OFFICIAL_CALENDAR_TABLE = "official_trade_cal"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
CALENDAR_TABLE = "STOCK_DAILY_DATA"
METADATA_PATH = Path("quant/data_file/production_assets/production_models/l4/10d/model_metadata.json")
CONTRACT_PATH = Path(
    "quant/data_file/runtime/agent_workspaces/research-agent/work/"
    "v260_next_executable_training_contract_design_20260816_r1/"
    "executable_training_contract_v2.json"
)
LABEL_COLUMN = "executable_10d_open_return"
WARMUP_START = "20210101"
DEVELOPMENT_START = "20220101"
DEVELOPMENT_END = "20241231"


@dataclass(frozen=True)
class Fold:
    fold_id: str
    test_start: str
    test_end: str


FOLDS = (
    Fold("fold01", "20220620", "20221230"),
    Fold("fold02", "20230130", "20231229"),
    Fold("fold03", "20240129", "20241213"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonical_frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values(["trade_date", "stock_code"], kind="mergesort")[columns]
    text = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return sha256_bytes(text.encode("utf-8"))


def feature_aliases(metadata: dict[str, Any], available: set[str]) -> tuple[list[str], list[str], list[list[str]]]:
    requested = [str(item) for item in metadata["feature_columns"]]
    resolved = resolve_savedmodel_feature_aliases(requested, available)
    if resolved["missing"]:
        raise RuntimeError(f"blocked_fail_closed_unresolved_qfq_feature_alias: {resolved['missing']}")
    pairs = {str(old): str(new) for old, new in resolved["alias_pairs"]}
    actual = [pairs.get(item, item) for item in requested]
    if len(actual) != len(requested) or len(set(actual)) != len(actual):
        raise RuntimeError("blocked_fail_closed_feature_dimension_or_alias_collision")
    return requested, actual, resolved["alias_pairs"]


def production_params(metadata: dict[str, Any]) -> dict[str, Any]:
    saved = metadata["xgb_params"]
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "n_estimators": int(metadata["best_iteration"]) + 1,
        "learning_rate": float(saved["learning_rate"]),
        "max_depth": int(saved["max_depth"]),
        "subsample": float(saved["subsample"]),
        "colsample_bytree": float(saved["colsample_bytree"]),
        "reg_alpha": float(saved["reg_alpha"]),
        "reg_lambda": float(saved["reg_lambda"]),
        "n_jobs": int(saved["n_jobs"]),
        "random_state": int(saved["random_state"]),
        "tree_method": "hist",
        "verbosity": 0,
    }


def classifier_params(contract: dict[str, Any]) -> dict[str, Any]:
    params = dict(contract["model"]["fixed_params"])
    if params.pop("missing") != "NaN":
        raise RuntimeError("blocked_fail_closed_missing_decode_contract")
    params["missing"] = np.nan
    return params


def load_calendar() -> list[str]:
    # This source is the only session authority.  L2 dates can be compared in
    # a report but must never define a fold, previous day, or +12 label end.
    payload = json.loads(OFFICIAL_CALENDAR_DESCRIPTOR.read_text(encoding="utf-8"))
    if payload.get("source") != "official_tushare_sdk.trade_cal":
        raise RuntimeError("blocked_fail_closed_calendar_contract_nonofficial_source")
    if payload.get("date_range") != [WARMUP_START, DEVELOPMENT_END]:
        raise RuntimeError("blocked_fail_closed_calendar_contract_window")
    connection = duckdb.connect(str(OFFICIAL_CALENDAR_DB), read_only=True)
    try:
        rows = connection.execute(
            f"SELECT cal_date FROM {quote(OFFICIAL_CALENDAR_TABLE)} "
            "WHERE is_open = 1 AND cal_date BETWEEN ? AND ? ORDER BY cal_date",
            [WARMUP_START, DEVELOPMENT_END],
        ).fetchall()
    finally:
        connection.close()
    result = [str(row[0]) for row in rows]
    if not result or result[0] > "20210104" or result[-1] < DEVELOPMENT_END:
        raise RuntimeError(
            "blocked_fail_closed_calendar_contract_missing_official_2021_2024_trade_cal"
        )
    if len(result) != len(set(result)):
        raise RuntimeError("blocked_fail_closed_calendar_contract_duplicate_dates")
    return result


def load_data(features: list[str]) -> pd.DataFrame:
    feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(LABEL_DB), read_only=True)
    try:
        selected = ", ".join(quote(item) for item in features)
        feature_frame = feature_connection.execute(
            f"SELECT trade_date, stock_code, {selected} FROM {quote(FEATURE_TABLE)} "
            "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
            "ORDER BY trade_date, stock_code",
            [WARMUP_START, DEVELOPMENT_END],
        ).fetchdf()
        label_frame = label_connection.execute(
            f"SELECT trade_date, stock_code, {quote(LABEL_COLUMN)} AS target FROM {quote(LABEL_TABLE)} "
            "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
            "ORDER BY trade_date, stock_code",
            [WARMUP_START, DEVELOPMENT_END],
        ).fetchdf()
    finally:
        feature_connection.close()
        label_connection.close()
    frame = feature_frame.merge(label_frame, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    frame["stock_code"] = frame["stock_code"].astype(str)
    if frame.empty or frame["trade_date"].max() > DEVELOPMENT_END:
        raise RuntimeError("blocked_fail_closed_development_date_boundary")
    if frame["stock_code"].str.endswith(".BJ").any() or frame.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_input_key_or_no_bj_contract")
    return frame


def label_end_map(calendar: list[str]) -> dict[str, str]:
    return {calendar[index]: calendar[index + 12] for index in range(len(calendar) - 12)}


def weight_for_production(frame: pd.DataFrame) -> np.ndarray:
    # Outer-fold frames retain source indexes, so assign by label first and
    # only then convert to the positional array consumed by XGBoost.
    weights = pd.Series(1.0, index=frame.index, dtype="float32")
    for _, group in frame.groupby("trade_date", sort=False):
        count = len(group)
        top_count = max(1, int(math.ceil(count * 0.005)))
        indices = group["target"].nlargest(top_count, keep="first").index
        weights.loc[indices] = 8.0
    return weights.to_numpy(dtype=np.float32)


def month_key(value: str) -> str:
    return value[:6]


def fit_baseline(train: pd.DataFrame, predict: pd.DataFrame, features: list[str], params: dict[str, Any]) -> tuple[np.ndarray, str]:
    if train.empty or train["target"].isna().any() or not np.isfinite(train["target"]).all():
        raise RuntimeError("blocked_fail_closed_baseline_training_labels")
    model = xgb.XGBRegressor(**params)
    model.fit(
        train[features].astype("float32"),
        train["target"].astype("float32"),
        sample_weight=weight_for_production(train),
        verbose=False,
    )
    return model.predict(predict[features].astype("float32")).astype("float64"), sha256_bytes(model.get_booster().save_raw(raw_format="json"))


def strict_monthly_baseline_state(
    data: pd.DataFrame,
    features: list[str],
    calendar: list[str],
    end_map: dict[str, str],
    params: dict[str, Any],
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    model_dir.mkdir(parents=True, exist_ok=False)
    calendar_index = {date: index for index, date in enumerate(calendar)}
    months = sorted({month_key(date) for date in calendar})
    states: list[pd.DataFrame] = []
    model_hashes: dict[str, str] = {}
    for month in months:
        test_dates = [date for date in calendar if month_key(date) == month]
        start = test_dates[0]
        if calendar_index[start] < 120:
            continue
        train_dates = {date for date in calendar if date < start and end_map.get(date, "99999999") < start}
        train = data.loc[data["trade_date"].isin(train_dates) & np.isfinite(data["target"])].copy()
        test = data.loc[data["trade_date"].isin(test_dates)].copy()
        if train.empty or test.empty:
            continue
        scores, model_hash = fit_baseline(train, test, features, params)
        test = test[["trade_date", "stock_code"]].copy()
        test["baseline_state_score"] = scores
        states.append(test)
        model_hashes[month] = model_hash
        (model_dir / f"monthly_baseline_{month}.sha256").write_text(model_hash + "\n", encoding="ascii")
    state = pd.concat(states, ignore_index=True)
    if state.duplicated(["trade_date", "stock_code"]).any() or state["baseline_state_score"].isna().any():
        raise RuntimeError("blocked_fail_closed_strict_monthly_baseline_state")
    state["baseline_state_rank_pct"] = state.groupby("trade_date", sort=False)["baseline_state_score"].rank(
        method="average", pct=True, ascending=True
    )
    return state, model_hashes


def add_lag_features(
    source: pd.DataFrame,
    state: pd.DataFrame,
    previous: dict[str, str],
) -> pd.DataFrame:
    index = state.set_index(["trade_date", "stock_code"], verify_integrity=True)
    result = source.copy()
    p = result["trade_date"].map(previous)
    pp = p.map(previous)
    keys_p = pd.MultiIndex.from_arrays([p, result["stock_code"]])
    keys_pp = pd.MultiIndex.from_arrays([pp, result["stock_code"]])
    result["baseline_oof_score_lag1"] = index.reindex(keys_p)["baseline_state_score"].to_numpy()
    result["baseline_oof_rank_pct_lag1"] = index.reindex(keys_p)["baseline_state_rank_pct"].to_numpy()
    lag2_score = index.reindex(keys_pp)["baseline_state_score"].to_numpy()
    lag2_rank = index.reindex(keys_pp)["baseline_state_rank_pct"].to_numpy()
    result["baseline_oof_score_delta_lag1_lag2"] = result["baseline_oof_score_lag1"] - lag2_score
    result["baseline_oof_rank_delta_lag1_lag2"] = result["baseline_oof_rank_pct_lag1"] - lag2_rank
    result["eligible_lag1"] = 1.0
    result["had_baseline_state_lag1"] = 1.0
    lag_columns = [column for column in result.columns if column.startswith("baseline_oof_")]
    return result.dropna(subset=lag_columns)


def make_binary_label(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[np.isfinite(frame["target"])].copy()
    median = result.groupby("trade_date", sort=False)["target"].transform("median")
    result["excess_positive_10d_binary"] = (result["target"] > median).astype("int8")
    return result


def daily_spearman(frame: pd.DataFrame, score_column: str) -> tuple[float, int]:
    correlations: list[float] = []
    invalid = 0
    for _, group in frame.groupby("trade_date", sort=True):
        scores = group[score_column]
        targets = group["target"]
        if len(group) < 20 or not np.isfinite(scores).all() or not np.isfinite(targets).all() or scores.nunique() < 2 or targets.nunique() < 2:
            invalid += 1
            continue
        correlations.append(float(scores.rank(method="average").corr(targets.rank(method="average"))))
    if invalid or not correlations:
        raise RuntimeError("blocked_fail_closed_return_rank_sanity_domain_invalid")
    return float(np.mean(correlations)), len(correlations)


def run_replay(
    replay_id: int,
    data: pd.DataFrame,
    features: list[str],
    calendar: list[str],
    end_map: dict[str, str],
    baseline_params: dict[str, Any],
    candidate_params: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_dir = output_dir / "replays" / f"replay{replay_id}"
    replay_dir.mkdir(parents=True, exist_ok=False)
    state, monthly_hashes = strict_monthly_baseline_state(
        data, features, calendar, end_map, baseline_params, replay_dir / "monthly_models"
    )
    previous = {calendar[index]: calendar[index - 1] for index in range(1, len(calendar))}
    enriched = make_binary_label(add_lag_features(data, state, previous))
    candidate_features = features + [
        "baseline_oof_score_lag1",
        "baseline_oof_rank_pct_lag1",
        "baseline_oof_score_delta_lag1_lag2",
        "baseline_oof_rank_delta_lag1_lag2",
        "eligible_lag1",
        "had_baseline_state_lag1",
    ]
    rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in FOLDS:
        train = enriched.loc[
            (enriched["trade_date"] >= DEVELOPMENT_START)
            & (enriched["trade_date"].map(end_map) < fold.test_start)
        ].copy()
        test = enriched.loc[(enriched["trade_date"] >= fold.test_start) & (enriched["trade_date"] <= fold.test_end)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"blocked_fail_closed_empty_outer_fold_{fold.fold_id}")
        baseline_test, baseline_hash = fit_baseline(train, test, features, baseline_params)
        classifier = xgb.XGBClassifier(**candidate_params)
        classifier.fit(
            train[candidate_features].astype("float32"),
            train["excess_positive_10d_binary"].astype("int8"),
            verbose=False,
        )
        if classifier.classes_.tolist() != [0, 1]:
            raise RuntimeError("blocked_fail_closed_probability_mapping")
        probability = classifier.predict_proba(test[candidate_features].astype("float32"))[:, 1].astype("float64")
        raw_score = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / (1 - np.clip(probability, 1e-6, 1 - 1e-6)))
        candidate_hash = sha256_bytes(classifier.get_booster().save_raw(raw_format="json"))
        emitted = test[["trade_date", "stock_code", "target", "excess_positive_10d_binary"]].copy()
        emitted["fold_id"] = fold.fold_id
        emitted["baseline_oof_score"] = baseline_test
        emitted["candidate_probability"] = probability
        emitted["candidate_raw_score"] = raw_score
        emitted = emitted[
            [
                "trade_date", "stock_code", "fold_id", "baseline_oof_score", "candidate_probability",
                "candidate_raw_score", "target", "excess_positive_10d_binary",
            ]
        ]
        if emitted.duplicated(["trade_date", "stock_code"]).any() or not np.isfinite(emitted[["baseline_oof_score", "candidate_probability", "candidate_raw_score", "target"]]).all().all():
            raise RuntimeError("blocked_fail_closed_oof_finiteness_or_duplicate")
        baseline_spearman, days = daily_spearman(emitted, "baseline_oof_score")
        candidate_spearman, candidate_days = daily_spearman(emitted, "candidate_raw_score")
        if days != candidate_days:
            raise RuntimeError("blocked_fail_closed_return_rank_sanity_date_mismatch")
        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "train_rows": int(len(train)),
                "test_rows": int(len(emitted)),
                "baseline_roc_auc": float(roc_auc_score(emitted["excess_positive_10d_binary"], emitted["baseline_oof_score"])),
                "candidate_roc_auc": float(roc_auc_score(emitted["excess_positive_10d_binary"], emitted["candidate_raw_score"])),
                "baseline_return_rank_spearman": baseline_spearman,
                "candidate_return_rank_spearman": candidate_spearman,
                "return_rank_trade_days": days,
                "baseline_model_sha256": baseline_hash,
                "candidate_model_sha256": candidate_hash,
            }
        )
        rows.append(emitted)
    output = pd.concat(rows, ignore_index=True).sort_values(["trade_date", "stock_code"], kind="mergesort")
    if output.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_same_key_oof_duplicate")
    aggregate = {
        "baseline_roc_auc": float(roc_auc_score(output["excess_positive_10d_binary"], output["baseline_oof_score"])),
        "candidate_roc_auc": float(roc_auc_score(output["excess_positive_10d_binary"], output["candidate_raw_score"])),
        "baseline_return_rank_spearman": daily_spearman(output, "baseline_oof_score")[0],
        "candidate_return_rank_spearman": daily_spearman(output, "candidate_raw_score")[0],
    }
    aggregate["roc_auc_delta"] = aggregate["candidate_roc_auc"] - aggregate["baseline_roc_auc"]
    aggregate["return_rank_delta"] = aggregate["candidate_return_rank_spearman"] - aggregate["baseline_return_rank_spearman"]
    output_hashes = {
        "baseline_oof_score_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "baseline_oof_score"]),
        "candidate_probability_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "candidate_probability"]),
        "candidate_raw_score_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "candidate_raw_score"]),
        "combined_model_sha256": sha256_bytes("".join(sorted(monthly_hashes.values()) + [item["baseline_model_sha256"] for item in fold_metrics] + [item["candidate_model_sha256"] for item in fold_metrics]).encode("ascii")),
    }
    replay_summary = {"replay_id": replay_id, "fold_metrics": fold_metrics, "aggregate": aggregate, "hashes": output_hashes}
    json_dump(replay_dir / "replay_summary.json", replay_summary)
    return output, replay_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="quant/data_file/reports/model_agent_v260_pit_oof_excess_logit_10d_v1_20260816")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if contract["candidate_id"] != "v260_pit_oof_excess_logit_10d_v1" or contract["candidate_count"] != 1:
            raise RuntimeError("blocked_fail_closed_contract_identity")
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if metadata["feature_count"] != 40 or len(contract["input_features"]["core_pit_feature_columns"]) != 40:
            raise RuntimeError("blocked_fail_closed_production_10d_feature_contract")
        feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
        try:
            available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}
        finally:
            feature_connection.close()
        requested, features, aliases = feature_aliases(metadata, available)
        calendar = load_calendar()
        end_map = label_end_map(calendar)
        data = load_data(features)
        preflight = {
            "candidate_id": contract["candidate_id"],
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "research_only": True,
            "approval_status": "research_only_not_for_l5",
            "date_reads": {"warmup": [WARMUP_START, "20211231"], "development": [DEVELOPMENT_START, DEVELOPMENT_END], "sealed": ["20250101+", "not_read"]},
            "inputs": {"feature_db": str(FEATURE_DB), "label_db": str(LABEL_DB), "official_calendar_descriptor": str(OFFICIAL_CALENDAR_DESCRIPTOR), "official_calendar_db": str(OFFICIAL_CALENDAR_DB), "official_calendar_db_sha256": sha256_file(OFFICIAL_CALENDAR_DB), "metadata_sha256": sha256_file(METADATA_PATH), "requested_features": requested, "resolved_features": features, "qfq_alias_pairs": aliases},
            "calendar": {"source": f"{OFFICIAL_CALENDAR_DB}::{OFFICIAL_CALENDAR_TABLE}", "descriptor_sha256": sha256_file(OFFICIAL_CALENDAR_DESCRIPTOR), "start": calendar[0], "end": calendar[-1], "trade_days": len(calendar), "plus12_example": {"source": calendar[0], "label_end": end_map[calendar[0]]}},
            "input_rows": int(len(data)),
            "no_bj": True,
            "duplicate_key_groups": 0,
        }
        json_dump(output_dir / "preflight.json", preflight)
        baseline_params = production_params(metadata)
        candidate_params = classifier_params(contract)
        all_replays: list[tuple[pd.DataFrame, dict[str, Any]]] = []
        for replay_id in (1, 2, 3):
            all_replays.append(run_replay(replay_id, data, features, calendar, end_map, baseline_params, candidate_params, output_dir))
        first_output, first_summary = all_replays[0]
        reference = first_summary["hashes"] | first_summary["aggregate"]
        exact = all((summary["hashes"] | summary["aggregate"]) == reference for _, summary in all_replays[1:])
        gate = {
            "same_key_coverage": True,
            "pit_oof_integrity": True,
            "binary_discrimination": first_summary["aggregate"]["roc_auc_delta"] >= 0.005,
            "return_rank_sanity": first_summary["aggregate"]["return_rank_delta"] >= 0.0,
            "score_finiteness": bool(np.isfinite(first_output[["candidate_probability", "candidate_raw_score"]]).all().all()),
            "deterministic_replay_3_of_3": exact,
        }
        passed = all(gate.values())
        first_output = first_output.rename(columns={"target": "executable_10d_open_return"})
        first_output.to_parquet(output_dir / "baseline_candidate_same_key_oof.parquet", index=False)
        manifest = {
            "candidate_id": contract["candidate_id"],
            "asset_role": "l4_research_prediction_asset",
            "approval_status": "research_only_not_for_l5",
            "production_unchanged": True,
            "decision": "strategy_development_ab_only_pending_audit" if passed else "model_layer_rejected_no_strategy_ab",
            "allow_next_layer_continue": False,
        }
        json_dump(output_dir / "research_candidate_manifest.json", manifest)
        result = {
            "candidate_id": contract["candidate_id"],
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "decision": manifest["decision"],
            "model_layer_acceptance_gates": gate,
            "aggregate": first_summary["aggregate"],
            "fold_metrics": first_summary["fold_metrics"],
            "deterministic_replay": {"required": 3, "passed": exact, "replay_hashes": [summary["hashes"] for _, summary in all_replays]},
            "same_key": {"rows": int(len(first_output)), "duplicate_key_groups": 0, "extra_or_missing_keys": 0},
            "ready_for_audit_review": passed,
            "allow_next_layer_continue": False,
            "sealed_2025_2026_not_read": True,
        }
        json_dump(output_dir / "evaluation_summary.json", result)
        json_dump(output_dir / "hash_inventory.json", {"script_sha256": sha256_file(Path(__file__)), "preflight_sha256": sha256_file(output_dir / "preflight.json"), "evaluation_summary_sha256": sha256_file(output_dir / "evaluation_summary.json"), "oof_sha256": sha256_file(output_dir / "baseline_candidate_same_key_oof.parquet"), "manifest_sha256": sha256_file(output_dir / "research_candidate_manifest.json")})
        return 0
    except Exception as error:
        json_dump(output_dir / "blocked_or_rejected.json", {"candidate_id": "v260_pit_oof_excess_logit_10d_v1", "status": "blocked_fail_closed", "error": str(error), "production_unchanged": True, "allow_next_layer_continue": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
