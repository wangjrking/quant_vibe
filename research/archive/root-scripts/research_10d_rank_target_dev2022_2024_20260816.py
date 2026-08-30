"""Strict 2022-2024, research-only 10D rank-target improvement build.

This deliberately has one degree of freedom: the candidate replaces the
fold-local regression target with the target's same-day cross-sectional rank.
Features, model capacity, weights, folds, and evaluation remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

import research_10d_ranking_improvement_dev2022_2024_20260816 as base


OUTPUT_DEFAULT = "quant/data_file/reports/model_agent_10d_rank_target_dev2022_2024_20260816"
SIZE_COLUMN_CANDIDATES = ("total_mv", "circ_mv", "float_mv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], weights: np.ndarray, target: pd.Series, params: dict[str, Any]) -> np.ndarray:
    valid = target.notna().to_numpy()
    if not valid.any():
        raise RuntimeError("all rank-target training labels are null")
    model = xgb.XGBRegressor(**params)
    model.fit(
        train.loc[valid, features].astype("float32"),
        target.loc[valid].astype("float32"),
        sample_weight=weights[valid],
        verbose=False,
    )
    return model.predict(test[features].astype("float32"))


def rank_target(frame: pd.DataFrame) -> pd.Series:
    # The target is computed within each observed train day only.  No later
    # date, future label, or CV test label contributes to this transformation.
    return frame.groupby("trade_date", sort=False)["target"].rank(method="average", pct=True)


def enrichment_frame(
    feature_connection: duckdb.DuckDBPyConnection,
    base_frame: pd.DataFrame,
    size_column: str,
) -> pd.DataFrame:
    dates = sorted(base_frame["trade_date"].unique().tolist())
    if not dates or dates[-1] > base.DEVELOPMENT_CLOSED_AFTER:
        raise RuntimeError("diagnostic attempted to access sealed period")
    data = feature_connection.execute(
        f"SELECT trade_date, stock_code, amount, {base.quote(size_column)}, index_2000_close "
        f"FROM {base.quote(base.FEATURE_TABLE)} "
        "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
        "ORDER BY trade_date, stock_code",
        [dates[0], dates[-1]],
    ).fetchdf()
    data["trade_date"] = data["trade_date"].astype(str)
    data["stock_code"] = data["stock_code"].astype(str)
    return base_frame.merge(data, on=["trade_date", "stock_code"], how="left", validate="one_to_one")


def attribution(predictions: pd.DataFrame, feature_connection: duckdb.DuckDBPyConnection, size_column: str) -> dict[str, Any]:
    frame = enrichment_frame(feature_connection, predictions, size_column).dropna(subset=["target", "pred_prob"])
    frame["year"] = frame["trade_date"].str.slice(0, 4)
    frame["score_decile"] = frame.groupby("trade_date", sort=False)["pred_prob"].rank(method="first", pct=True)
    frame["score_decile"] = np.minimum(10, np.ceil(frame["score_decile"] * 10)).astype(int)
    liquidity_rank = frame.groupby("trade_date", sort=False)["amount"].rank(method="first", pct=True)
    size_rank = frame.groupby("trade_date", sort=False)[size_column].rank(method="first", pct=True)
    # Source-limited size/amount values are a diagnostic category, not an
    # imputed number.  This keeps attribution fail-transparent.
    frame["liquidity_tertile"] = np.minimum(3, np.ceil(liquidity_rank * 3)).fillna(0).astype(int)
    frame["size_tertile"] = np.minimum(3, np.ceil(size_rank * 3)).fillna(0).astype(int)

    closes = frame.groupby("trade_date", sort=True)["index_2000_close"].first().sort_index()
    trailing_return = closes.pct_change(20)
    market_state = pd.Series(np.where(trailing_return >= 0, "index_up_20d", "index_down_20d"), index=closes.index)
    frame["market_state"] = frame["trade_date"].map(market_state.to_dict())

    def grouped_summary(columns: list[str]) -> list[dict[str, Any]]:
        records = []
        for keys, group in frame.groupby(columns, sort=True, dropna=False):
            keys = (keys,) if not isinstance(keys, tuple) else keys
            records.append(
                dict(zip(columns, map(str, keys)))
                | {
                    "rows": int(len(group)),
                    "mean_target": float(group["target"].mean()),
                    "prediction_target_pearson": base.safe_corr(group["pred_prob"], group["target"]),
                }
            )
        return records

    daily = []
    top_sets: dict[str, set[str]] = {}
    for date, group in frame.groupby("trade_date", sort=True):
        rank_ic = base.safe_corr(group["pred_prob"].rank(method="average"), group["target"].rank(method="average"))
        ranked = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        top10 = ranked.head(10)
        daily.append(
            {
                "trade_date": str(date),
                "year": str(date)[:4],
                "rank_ic": rank_ic,
                "top10_excess": float(top10["target"].mean() - ranked["target"].mean()),
            }
        )
        top_sets[str(date)] = set(top10["stock_code"])
    daily_frame = pd.DataFrame(daily)
    yearly = []
    for year, group in daily_frame.groupby("year", sort=True):
        dates = group["trade_date"].tolist()
        overlaps = [len(top_sets[left] & top_sets[right]) / 10 for left, right in zip(dates, dates[1:])]
        yearly.append(
            {
                "year": year,
                "trade_days": int(len(group)),
                "rank_ic_mean": float(group["rank_ic"].mean()),
                "rank_ic_positive_day_share": float((group["rank_ic"] > 0).mean()),
                "top10_excess_mean": float(group["top10_excess"].mean()),
                "top10_turnover_proxy": None if not overlaps else float(1 - np.mean(overlaps)),
            }
        )
    return {
        "size_column": size_column,
        "yearly": yearly,
        "by_score_decile": grouped_summary(["year", "score_decile"]),
        "by_size_tertile": grouped_summary(["year", "size_tertile"]),
        "by_liquidity_tertile": grouped_summary(["year", "liquidity_tertile"]),
        "by_market_state": grouped_summary(["year", "market_state"]),
    }


def core(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    return {key: metrics[key] for key in (
        "evaluable_trade_days", "rank_ic_mean", "rank_ic_positive_day_share",
        "top_decile_excess_mean", "top10_excess_mean", "calibration_decile_corr_mean", "top10_turnover_proxy",
    )}


def finalize_from_checkpoints(output_dir: Path) -> int:
    """Finish an evidence-only remediation without recomputing model scores."""
    required = [output_dir / f"fold0{index}_summary.json" for index in (1, 2, 3)]
    if any(not path.exists() for path in required):
        raise RuntimeError("cannot finalize: fold model-score checkpoints are incomplete")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in required]
    prior_oof = Path(
        "quant/data_file/reports/model_agent_10d_ranking_improvement_dev2022_2024_20260816/oof_predictions.parquet"
    )
    if not prior_oof.exists():
        raise RuntimeError("cannot finalize: same-protocol baseline OOF checkpoint is missing")
    baseline = pd.read_parquet(prior_oof)
    baseline = baseline.loc[baseline["variant"] == "baseline"].copy()
    baseline["trade_date"] = baseline["trade_date"].astype(str)
    expected_folds = {record["fold"]["fold_id"] for record in results}
    if baseline.empty or set(baseline["fold_id"].unique()) != expected_folds:
        raise RuntimeError("cannot finalize: baseline OOF checkpoint fold domain differs")
    if baseline["trade_date"].max() > base.DEVELOPMENT_CLOSED_AFTER:
        raise RuntimeError("cannot finalize: baseline OOF checkpoint includes sealed period")
    feature_connection = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    try:
        available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({base.quote(base.FEATURE_TABLE)})").fetchall()}
        size_column = next((name for name in SIZE_COLUMN_CANDIDATES if name in available), None)
        if size_column is None:
            raise RuntimeError("cannot finalize: no size attribution column")
        diagnostic = attribution(baseline, feature_connection, size_column)
    finally:
        feature_connection.close()
    dump_json(output_dir / "production_spec_error_attribution.json", diagnostic)
    aggregate = {key: float(np.mean([record["delta"][key] for record in results])) for key in results[0]["delta"]}
    passed = (
        aggregate["rank_ic_mean"] > 0
        and aggregate["top_decile_excess_mean"] > 0
        and aggregate["top10_excess_mean"] > 0
        and sum(record["delta"]["rank_ic_mean"] > 0 for record in results) >= 2
        and sum(record["delta"]["top_decile_excess_mean"] > 0 for record in results) >= 2
    )
    summary = {
        "asset_role": "research_only_l4_candidate_evaluation",
        "approval_status": "research_only_not_for_l5",
        "development_closed_after": base.DEVELOPMENT_CLOSED_AFTER,
        "fold_results": results,
        "aggregate_delta_candidate_minus_baseline": aggregate,
        "predeclared_acceptance_gate": {"passed": passed},
        "decision": "freeze_unique_research_candidate_for_strategy_development_ab" if passed else "reject_no_further_search",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "production_unchanged": True,
        "metadata_only_finalization": True,
    }
    dump_json(output_dir / "evaluation_summary.json", summary)
    dump_json(output_dir / "metadata_only_remediation.json", {
        "reason": "all three fold score checkpoints completed; only size/amount null categorization blocked post-fit attribution",
        "model_scores_recomputed": False,
        "candidate_fold_summaries_unchanged": True,
        "baseline_oof_source": str(prior_oof),
        "baseline_oof_sha256": sha256_file(prior_oof),
        "development_closed_after": base.DEVELOPMENT_CLOSED_AFTER,
    })
    dump_json(output_dir / "hash_inventory.json", {
        "preflight_sha256": sha256_file(output_dir / "preflight.json"),
        "fold01_sha256": sha256_file(required[0]),
        "fold02_sha256": sha256_file(required[1]),
        "fold03_sha256": sha256_file(required[2]),
        "attribution_sha256": sha256_file(output_dir / "production_spec_error_attribution.json"),
        "evaluation_summary_sha256": sha256_file(output_dir / "evaluation_summary.json"),
        "metadata_only_remediation_sha256": sha256_file(output_dir / "metadata_only_remediation.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return 0


def materialize_candidate_oof(output_dir: Path) -> int:
    """Deterministically persist the already-frozen sole candidate's OOF scores."""
    summary_path = output_dir / "evaluation_summary.json"
    if not summary_path.exists():
        raise RuntimeError("cannot materialize: evaluation summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("decision") != "freeze_unique_research_candidate_for_strategy_development_ab":
        raise RuntimeError("cannot materialize: sole candidate did not pass frozen gate")
    destination = output_dir / "rank_target_candidate_oof_predictions.parquet"
    if destination.exists():
        raise RuntimeError("refusing to overwrite frozen candidate OOF asset")
    metadata = base.load_metadata()
    feature_connection = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(base.LABEL_DB), read_only=True)
    try:
        available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({base.quote(base.FEATURE_TABLE)})").fetchall()}
        aliases = base.resolve_savedmodel_feature_aliases([str(item) for item in metadata["feature_columns"]], available)
        if aliases["missing"]:
            raise RuntimeError(f"cannot materialize: production aliases changed: {aliases['missing']}")
        alias_lookup = dict(aliases["alias_pairs"])
        features = [alias_lookup.get(str(item), str(item)) for item in metadata["feature_columns"]]
        params = base.xgb_parameters(metadata)
        output: list[pd.DataFrame] = []
        checks: list[dict[str, Any]] = []
        expected_by_fold = {item["fold"]["fold_id"]: item["candidate"] for item in summary["fold_results"]}
        for fold in base.FOLDS:
            train = base.query_fold_frame(feature_connection, label_connection, features, fold.train_start, fold.train_end)
            test = base.query_fold_frame(feature_connection, label_connection, features, fold.test_start, fold.test_end)
            pred = fit_predict(train, test, features, base.production_weights(train), rank_target(train), params)
            frame = test[["trade_date", "stock_code", "target"]].copy()
            frame["pred_prob"] = pred.astype("float64")
            metrics = core(base.evaluate(frame))
            expected = expected_by_fold[fold.fold_id]
            differences = {key: abs(float(metrics[key]) - float(expected[key])) for key in expected if metrics[key] is not None}
            if any(value > 1e-12 for value in differences.values()):
                raise RuntimeError(f"{fold.fold_id}: candidate OOF replay does not match frozen checkpoint")
            frame["fold_id"] = fold.fold_id
            frame["variant"] = "rank_target_candidate"
            output.append(frame)
            checks.append({"fold_id": fold.fold_id, "metric_abs_differences": differences})
        candidate_oof = pd.concat(output, ignore_index=True)
        if candidate_oof["trade_date"].astype(str).max() > base.DEVELOPMENT_CLOSED_AFTER:
            raise RuntimeError("candidate OOF materialization read sealed data")
        if candidate_oof.duplicated(["trade_date", "stock_code"]).any() or candidate_oof["pred_prob"].isna().any():
            raise RuntimeError("candidate OOF quality gate failed")
        candidate_oof.to_parquet(destination, index=False)
        manifest = {
            "asset_role": "l4_research_prediction_asset",
            "approval_status": "research_only_not_for_l5",
            "candidate_id": "research_10d_rank_target_dev2022_2024_20260816",
            "frozen": True,
            "development_closed_after": base.DEVELOPMENT_CLOSED_AFTER,
            "sealed_periods_not_read": ["2025", "2026+"],
            "method": "production 10D XGBoost specification with same-day cross-sectional rank target; no hyperparameter change",
            "features": 40,
            "folds": [fold.__dict__ for fold in base.FOLDS],
            "oof_asset": str(destination),
            "oof_sha256": sha256_file(destination),
            "rows": int(len(candidate_oof)),
            "min_trade_date": str(candidate_oof["trade_date"].min()),
            "max_trade_date": str(candidate_oof["trade_date"].max()),
            "duplicate_key_groups": 0,
            "null_pred_prob": 0,
            "no_bj": True,
            "replay_metric_checks": checks,
            "allow_for_strategy_development_ab_only": True,
            "approved_for_l5": False,
            "allow_next_layer_continue": False,
            "production_unchanged": True,
        }
        dump_json(output_dir / "research_candidate_manifest.json", manifest)
        inventory_path = output_dir / "hash_inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["candidate_oof_sha256"] = sha256_file(destination)
        inventory["candidate_manifest_sha256"] = sha256_file(output_dir / "research_candidate_manifest.json")
        dump_json(inventory_path, inventory)
        return 0
    finally:
        feature_connection.close()
        label_connection.close()


def probe_candidate_oof(output_dir: Path) -> int:
    path = output_dir / "rank_target_candidate_oof_predictions.parquet"
    if not path.exists():
        raise RuntimeError("candidate OOF asset is missing")
    frame = pd.read_parquet(path, columns=["trade_date", "stock_code", "pred_prob"])
    result = {
        "rows": int(len(frame)),
        "stocks": int(frame["stock_code"].nunique()),
        "min_trade_date": str(frame["trade_date"].min()),
        "max_trade_date": str(frame["trade_date"].max()),
        "duplicate_key_groups": int(frame.duplicated(["trade_date", "stock_code"]).sum()),
        "null_pred_prob": int(frame["pred_prob"].isna().sum()),
        "nonfinite_pred_prob": int((~np.isfinite(frame["pred_prob"])).sum()),
        "bj_rows": int(frame["stock_code"].astype(str).str.endswith(".BJ").sum()),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def refresh_hash_inventory(output_dir: Path) -> int:
    inventory_path = output_dir / "hash_inventory.json"
    if not inventory_path.exists():
        raise RuntimeError("hash inventory is missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    report_path = output_dir / "research_build_report.md"
    manifest_path = output_dir / "research_candidate_manifest.json"
    if report_path.exists():
        inventory["research_report_sha256"] = sha256_file(report_path)
    if manifest_path.exists():
        inventory["candidate_manifest_sha256"] = sha256_file(manifest_path)
    inventory["script_sha256"] = sha256_file(Path(__file__))
    inventory["hash_inventory_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(inventory_path, inventory)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    parser.add_argument("--finalize-from-checkpoints", action="store_true")
    parser.add_argument("--materialize-candidate-oof", action="store_true")
    parser.add_argument("--probe-candidate-oof", action="store_true")
    parser.add_argument("--refresh-hash-inventory", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.finalize_from_checkpoints:
        return finalize_from_checkpoints(output_dir)
    if args.materialize_candidate_oof:
        return materialize_candidate_oof(output_dir)
    if args.probe_candidate_oof:
        return probe_candidate_oof(output_dir)
    if args.refresh_hash_inventory:
        return refresh_hash_inventory(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite research output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    metadata = base.load_metadata()
    feature_connection = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(base.LABEL_DB), read_only=True)
    try:
        available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({base.quote(base.FEATURE_TABLE)})").fetchall()}
        size_column = next((name for name in SIZE_COLUMN_CANDIDATES if name in available), None)
        if size_column is None:
            raise RuntimeError("no approved size column exists for diagnostic attribution")
        requested = [str(item) for item in metadata["feature_columns"]]
        aliases = base.resolve_savedmodel_feature_aliases(requested, available)
        if aliases["missing"]:
            raise RuntimeError(f"unresolved production aliases: {aliases['missing']}")
        alias_lookup = dict(aliases["alias_pairs"])
        features = [alias_lookup.get(name, name) for name in requested]
        if len(features) != 40 or len(set(features)) != 40:
            raise RuntimeError("production feature contract changed")

        preflight = {
            "asset_role": "research_only_l4_candidate_build",
            "approval_status": "research_only_not_for_l5",
            "development_closed_after": base.DEVELOPMENT_CLOSED_AFTER,
            "sealed_periods_not_read": ["2025", "2026+"],
            "production_assets_unchanged": True,
            "baseline": "fold-local production_10d_xgb specification and frozen daily_top_quantile weights",
            "principal_candidate": "same features, capacity, folds and weights; train only on same-day cross-sectional rank of 10D label",
            "robustness_controls": [],
            "no_parameter_search": True,
            "inputs": {
                "feature_db": str(base.FEATURE_DB), "feature_db_sha256": sha256_file(base.FEATURE_DB),
                "label_db": str(base.LABEL_DB), "label_db_sha256": sha256_file(base.LABEL_DB),
                "production_metadata": str(base.METADATA_PATH), "production_metadata_sha256": sha256_file(base.METADATA_PATH),
                "qfq_alias_pairs": aliases["alias_pairs"], "size_column": size_column,
            },
            "folds": [fold.__dict__ | base.validate_fold_calendar(feature_connection, fold) for fold in base.FOLDS],
        }
        dump_json(output_dir / "preflight.json", preflight)

        params = base.xgb_parameters(metadata)
        results: list[dict[str, Any]] = []
        baseline_predictions: list[pd.DataFrame] = []
        all_predictions: list[pd.DataFrame] = []
        for fold in base.FOLDS:
            train = base.query_fold_frame(feature_connection, label_connection, features, fold.train_start, fold.train_end)
            test = base.query_fold_frame(feature_connection, label_connection, features, fold.test_start, fold.test_end)
            baseline_pred = fit_predict(train, test, features, base.production_weights(train), train["target"], params)
            candidate_pred = fit_predict(train, test, features, base.production_weights(train), rank_target(train), params)
            baseline_frame = test[["trade_date", "stock_code", "target"]].copy()
            baseline_frame["pred_prob"] = baseline_pred.astype("float64")
            candidate_frame = test[["trade_date", "stock_code", "target"]].copy()
            candidate_frame["pred_prob"] = candidate_pred.astype("float64")
            baseline_metrics = base.evaluate(baseline_frame)
            candidate_metrics = base.evaluate(candidate_frame)
            delta = {key: float(candidate_metrics[key] - baseline_metrics[key]) for key in (
                "rank_ic_mean", "top_decile_excess_mean", "top10_excess_mean", "calibration_decile_corr_mean", "top10_turnover_proxy",
            )}
            record = {"fold": fold.__dict__, "baseline": core(baseline_metrics), "candidate": core(candidate_metrics), "delta": delta}
            results.append(record)
            dump_json(output_dir / f"{fold.fold_id}_summary.json", record)
            baseline_frame["fold_id"] = fold.fold_id
            baseline_frame["variant"] = "production_spec_baseline"
            candidate_frame["fold_id"] = fold.fold_id
            candidate_frame["variant"] = "rank_target_candidate"
            baseline_predictions.append(baseline_frame.copy())
            all_predictions.extend((baseline_frame, candidate_frame))

        baseline_all = pd.concat(baseline_predictions, ignore_index=True)
        diagnostic = attribution(baseline_all, feature_connection, size_column)
        dump_json(output_dir / "production_spec_error_attribution.json", diagnostic)
        aggregate = {key: float(np.mean([record["delta"][key] for record in results])) for key in results[0]["delta"]}
        passed = (
            aggregate["rank_ic_mean"] > 0
            and aggregate["top_decile_excess_mean"] > 0
            and aggregate["top10_excess_mean"] > 0
            and sum(record["delta"]["rank_ic_mean"] > 0 for record in results) >= 2
            and sum(record["delta"]["top_decile_excess_mean"] > 0 for record in results) >= 2
        )
        summary = {
            "asset_role": "research_only_l4_candidate_evaluation",
            "approval_status": "research_only_not_for_l5",
            "development_closed_after": base.DEVELOPMENT_CLOSED_AFTER,
            "fold_results": results,
            "aggregate_delta_candidate_minus_baseline": aggregate,
            "predeclared_acceptance_gate": {"passed": passed},
            "decision": "freeze_unique_research_candidate_for_strategy_development_ab" if passed else "reject_no_further_search",
            "ready_for_audit_review": True,
            "allow_next_layer_continue": False,
            "production_unchanged": True,
        }
        dump_json(output_dir / "evaluation_summary.json", summary)
        pd.concat(all_predictions, ignore_index=True).to_parquet(output_dir / "oof_predictions.parquet", index=False)
        dump_json(output_dir / "hash_inventory.json", {
            "preflight_sha256": sha256_file(output_dir / "preflight.json"),
            "attribution_sha256": sha256_file(output_dir / "production_spec_error_attribution.json"),
            "evaluation_summary_sha256": sha256_file(output_dir / "evaluation_summary.json"),
            "oof_predictions_sha256": sha256_file(output_dir / "oof_predictions.parquet"),
            "script_sha256": sha256_file(Path(__file__)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        return 0
    finally:
        feature_connection.close()
        label_connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
