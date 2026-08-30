"""One frozen research-only 10D residual-ranking candidate.

The build is intentionally constrained by the strategy-to-model handoff:
one XGBRanker candidate, three anchored folds, 2022-2024 only, and no
production/strategy writes.  A fold-local production-spec regressor supplies
the base margin; the ranker can only learn an additive within-day residual.
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


OUTPUT_DEFAULT = "quant/data_file/reports/model_agent_10d_ndcg10_residual_boost_dev2022_2024_20260816"
CONTRACT_PATH = Path(
    "quant/data_file/runtime/agent_workspaces/strategy-agent/work/"
    "prod_v260_model_topk_stability_handoff_20260816/model_improvement_handoff_v1.json"
)
CONTRACT_SHA256 = "ba3fff2215dc13f3d8da060e4d757f5569742430b1ce9da28c624a8138dbf8ed"
CANDIDATE_ID = "date_grouped_ndcg10_residual_boost_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolved_features(metadata: dict[str, Any], available: set[str]) -> tuple[list[str], list[list[str]]]:
    requested = [str(item) for item in metadata["feature_columns"]]
    aliases = base.resolve_savedmodel_feature_aliases(requested, available)
    if aliases["missing"]:
        raise RuntimeError(f"unresolved production feature aliases: {aliases['missing']}")
    alias_lookup = dict(aliases["alias_pairs"])
    features = [alias_lookup.get(name, name) for name in requested]
    if len(features) != 40 or len(set(features)) != len(features):
        raise RuntimeError("production 10D feature contract drifted")
    return features, aliases["alias_pairs"]


def top10_relevance(frame: pd.DataFrame) -> np.ndarray:
    """Binary train-only relevance, deterministic under ties via stock_code."""
    ordered = frame.sort_values(
        ["trade_date", "target", "stock_code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    positions = ordered.groupby("trade_date", sort=False).cumcount()
    relevance = pd.Series(0, index=frame.index, dtype="int8")
    relevance.loc[ordered.index[positions < 10]] = 1
    return relevance.to_numpy(dtype=np.int8)


def ranker_params(metadata: dict[str, Any]) -> dict[str, Any]:
    params = base.xgb_parameters(metadata)
    params.update(
        {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@10",
            "lambdarank_pair_method": "topk",
            "lambdarank_num_pair_per_sample": 10,
        }
    )
    return params


def fit_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    weights: np.ndarray,
    params: dict[str, Any],
) -> tuple[xgb.XGBRegressor, np.ndarray, np.ndarray]:
    valid = train["target"].notna().to_numpy()
    if not valid.any():
        raise RuntimeError("baseline training labels are all null")
    model = xgb.XGBRegressor(**params)
    model.fit(
        train.loc[valid, features].astype("float32"),
        train.loc[valid, "target"].astype("float32"),
        sample_weight=weights[valid],
        verbose=False,
    )
    train_margin = model.predict(train[features].astype("float32"))
    test_margin = model.predict(test[features].astype("float32"))
    if not np.isfinite(train_margin).all() or not np.isfinite(test_margin).all():
        raise RuntimeError("production-spec base margin is nonfinite")
    return model, train_margin.astype("float32"), test_margin.astype("float32")


def fit_residual_ranker(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    train_margin: np.ndarray,
    test_margin: np.ndarray,
    params: dict[str, Any],
) -> tuple[xgb.XGBRanker, np.ndarray]:
    valid = train["target"].notna().to_numpy()
    train_valid = train.loc[valid].sort_values(["trade_date", "stock_code"], kind="mergesort")
    sorted_indices = train_valid.index.to_numpy()
    ordered_positions = train.index.get_indexer(sorted_indices)
    if (ordered_positions < 0).any():
        raise RuntimeError("ranker training index alignment failed")
    qid = pd.factorize(train_valid["trade_date"], sort=True)[0]
    relevance = top10_relevance(train_valid)
    if relevance.sum() < 10:
        raise RuntimeError("ranker relevance has no positive examples")
    # XGBRanker consumes one weight per query group, not one per stock row.
    # The frozen contract authorizes no stock/recent-date weighting rule, so
    # every training trade_date gets the same deterministic group weight.
    group_weights = np.ones(int(qid.max()) + 1, dtype=np.float32)
    model = xgb.XGBRanker(**params)
    model.fit(
        train_valid[features].astype("float32"),
        relevance,
        qid=qid,
        sample_weight=group_weights,
        base_margin=train_margin[ordered_positions],
        verbose=False,
    )
    prediction = model.predict(test[features].astype("float32"), base_margin=test_margin)
    if not np.isfinite(prediction).all():
        raise RuntimeError("candidate prediction is nonfinite")
    return model, prediction.astype("float64")


def core_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    keys = (
        "evaluable_trade_days",
        "rank_ic_mean",
        "rank_ic_positive_day_share",
        "top_decile_excess_mean",
        "top10_excess_mean",
        "calibration_decile_corr_mean",
        "top10_turnover_proxy",
    )
    return {key: metrics[key] for key in keys}


def delta_metrics(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = (
        "rank_ic_mean",
        "top_decile_excess_mean",
        "top10_excess_mean",
        "calibration_decile_corr_mean",
        "top10_turnover_proxy",
    )
    return {key: float(candidate[key] - baseline[key]) for key in keys}


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    aggregate = summary["aggregate_delta_candidate_minus_baseline"]
    lines = [
        "# 10D NDCG@10 残差排序研究构建",
        "",
        f"- 唯一候选：`{CANDIDATE_ID}`",
        "- 范围：2022-2024 三折 anchored CV；2025 与 2026 未读取。",
        "- 模型：生产 10D 折内基线为 base margin，XGBRanker 仅学习按交易日分组的加性排序残差。",
        f"- 决策：`{decision}`",
        f"- 聚合 RankIC 增量：`{aggregate['rank_ic_mean']:.8f}`",
        f"- 聚合 Top10 excess 增量：`{aggregate['top10_excess_mean']:.8f}`",
        f"- 聚合 Top10 turnover proxy 增量：`{aggregate['top10_turnover_proxy']:.8f}`",
        "- 本构建未运行策略 A/B、验证或生产发布；策略层验收仍须在其冻结执行器内独立完成。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the sole frozen 10D NDCG@10 residual candidate.")
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite research output directory: {output_dir}")
    if xgb.__version__ != "2.1.4":
        raise RuntimeError(f"contract requires xgboost 2.1.4, got {xgb.__version__}")
    if sha256_file(CONTRACT_PATH) != CONTRACT_SHA256:
        raise RuntimeError("frozen handoff contract SHA256 mismatch")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["next_candidate_contract"]["candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("handoff candidate id does not match the fixed implementation")
    if contract["evidence_scope"]["date_max"] != "20241231" or contract["allow_production_change"]:
        raise RuntimeError("handoff boundary drifted")
    output_dir.mkdir(parents=True)

    metadata = base.load_metadata()
    if metadata.get("sample_weight_config") != {
        "mode": "daily_top_quantile", "top_pct": base.TOP_PCT, "top_multiplier": base.TOP_MULTIPLIER,
    }:
        raise RuntimeError("production 10D weighting contract drifted")
    feature_connection = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(base.LABEL_DB), read_only=True)
    try:
        available = {str(row[1]) for row in feature_connection.execute(
            f"PRAGMA table_info({base.quote(base.FEATURE_TABLE)})"
        ).fetchall()}
        features, aliases = resolved_features(metadata, available)
        preflight = {
            "asset_role": "research_only_l4_candidate_build",
            "approval_status": "research_only_not_for_l5",
            "candidate_id": CANDIDATE_ID,
            "frozen_contract_path": str(CONTRACT_PATH),
            "frozen_contract_sha256": CONTRACT_SHA256,
            "development_dates_read": {"min": "20220104", "max": base.DEVELOPMENT_CLOSED_AFTER},
            "sealed_periods_not_read": ["2025", "2026+"],
            "production_assets_unchanged": True,
            "model_contract": {
                "xgboost_version": xgb.__version__,
                "estimator": "XGBRanker",
                "objective": "rank:ndcg",
                "eval_metric": "ndcg@10",
                "qid": "trade_date ordered ascending with stock_code tie ordering",
                "relevance": "per train trade_date realized 10D target top10=1, else=0",
                "base_margin": "fold-local production-spec baseline score; test rows remain OOF",
                "lambdarank_pair_method": "topk",
                "lambdarank_num_pair_per_sample": 10,
                "parameter_search": False,
                "turnover_penalty": False,
                "ranker_group_weighting": "equal one weight per trade_date; no stock-level or recency weighting",
            },
            "inputs": {
                "feature_db": str(base.FEATURE_DB), "feature_table": base.FEATURE_TABLE,
                "label_db": str(base.LABEL_DB), "label_table": base.LABEL_TABLE,
                "label_column": base.LABEL_COLUMN,
                "production_metadata": str(base.METADATA_PATH),
                "production_metadata_sha256": sha256_file(base.METADATA_PATH),
                "feature_columns": features,
                "qfq_alias_pairs": aliases,
            },
            "folds": [fold.__dict__ | base.validate_fold_calendar(feature_connection, fold) for fold in base.FOLDS],
        }
        dump_json(output_dir / "preflight.json", preflight)

        baseline_params = base.xgb_parameters(metadata)
        candidate_params = ranker_params(metadata)
        results: list[dict[str, Any]] = []
        all_predictions: list[pd.DataFrame] = []
        model_hashes: list[dict[str, Any]] = []
        for fold in base.FOLDS:
            train = base.query_fold_frame(feature_connection, label_connection, features, fold.train_start, fold.train_end)
            test = base.query_fold_frame(feature_connection, label_connection, features, fold.test_start, fold.test_end)
            if test["trade_date"].max() > base.DEVELOPMENT_CLOSED_AFTER:
                raise RuntimeError(f"{fold.fold_id}: attempted sealed-period read")
            weights = base.production_weights(train)
            baseline_model, train_margin, test_margin = fit_baseline(train, test, features, weights, baseline_params)
            ranker, candidate_pred = fit_residual_ranker(
                train, test, features, train_margin, test_margin, candidate_params
            )
            baseline_frame = test[["trade_date", "stock_code", "target"]].copy()
            baseline_frame["pred_prob"] = test_margin.astype("float64")
            candidate_frame = test[["trade_date", "stock_code", "target"]].copy()
            candidate_frame["pred_prob"] = candidate_pred
            if not baseline_frame[["trade_date", "stock_code"]].equals(candidate_frame[["trade_date", "stock_code"]]):
                raise RuntimeError(f"{fold.fold_id}: baseline/candidate key domain differs")
            baseline_metrics = base.evaluate(baseline_frame)
            candidate_metrics = base.evaluate(candidate_frame)

            model_dir = output_dir / "models"
            model_dir.mkdir(exist_ok=True)
            baseline_path = model_dir / f"{fold.fold_id}_production_spec_baseline.json"
            ranker_path = model_dir / f"{fold.fold_id}_ndcg10_residual_ranker.json"
            baseline_model.save_model(str(baseline_path))
            ranker.save_model(str(ranker_path))
            replay = xgb.XGBRanker()
            replay.load_model(str(ranker_path))
            replay_pred = replay.predict(test[features].astype("float32"), base_margin=test_margin)
            replay_max_abs = float(np.max(np.abs(replay_pred.astype("float64") - candidate_pred)))
            if replay_max_abs > 1e-12:
                raise RuntimeError(f"{fold.fold_id}: deterministic model replay mismatch {replay_max_abs}")
            record = {
                "fold": fold.__dict__,
                "train_rows": int(len(train)), "test_rows": int(len(test)),
                "baseline": core_metrics(baseline_metrics),
                "candidate": core_metrics(candidate_metrics),
                "delta": delta_metrics(candidate_metrics, baseline_metrics),
                "replay": {"passed": True, "max_abs_prediction_difference": replay_max_abs},
                "base_margin": {"train_kind": "fold_local_production_spec_score", "test_kind": "fold_local_oof_production_spec_score"},
                "model_paths": {"baseline": str(baseline_path), "candidate": str(ranker_path)},
            }
            results.append(record)
            dump_json(output_dir / f"{fold.fold_id}_summary.json", record)
            baseline_frame["fold_id"] = fold.fold_id
            baseline_frame["variant"] = "production_spec_baseline"
            candidate_frame["fold_id"] = fold.fold_id
            candidate_frame["variant"] = CANDIDATE_ID
            all_predictions.extend((baseline_frame, candidate_frame))
            model_hashes.append({
                "fold_id": fold.fold_id,
                "baseline_model_sha256": sha256_file(baseline_path),
                "candidate_model_sha256": sha256_file(ranker_path),
            })

        oof = pd.concat(all_predictions, ignore_index=True)
        oof["trade_date"] = oof["trade_date"].astype(str)
        oof["stock_code"] = oof["stock_code"].astype(str)
        quality = {
            "rows": int(len(oof)), "min_trade_date": str(oof["trade_date"].min()), "max_trade_date": str(oof["trade_date"].max()),
            "duplicate_key_groups": int(oof.duplicated(["fold_id", "variant", "trade_date", "stock_code"]).sum()),
            "null_pred_prob": int(oof["pred_prob"].isna().sum()),
            "nonfinite_pred_prob": int((~np.isfinite(oof["pred_prob"])).sum()),
            "bj_rows": int(oof["stock_code"].str.endswith(".BJ").sum()),
            "same_key_domain_by_fold": True,
        }
        if quality["max_trade_date"] > base.DEVELOPMENT_CLOSED_AFTER or any(
            quality[key] for key in ("duplicate_key_groups", "null_pred_prob", "nonfinite_pred_prob", "bj_rows")
        ):
            raise RuntimeError("OOF quality or sealed-period gate failed")
        oof_path = output_dir / "baseline_candidate_same_key_oof.parquet"
        oof.to_parquet(oof_path, index=False)

        aggregate = {key: float(np.mean([record["delta"][key] for record in results])) for key in results[0]["delta"]}
        gate = {
            "aggregate_rank_ic_delta_gte_zero": aggregate["rank_ic_mean"] >= 0.0,
            "aggregate_top10_excess_delta_gt_zero": aggregate["top10_excess_mean"] > 0.0,
            "each_fold_top10_excess_delta_gte_zero": all(record["delta"]["top10_excess_mean"] >= 0.0 for record in results),
            "aggregate_top10_turnover_proxy_delta_lte_zero": aggregate["top10_turnover_proxy"] <= 0.0,
            "deterministic_replay": all(record["replay"]["passed"] for record in results),
        }
        passed = all(gate.values())
        decision = "freeze_unique_research_candidate_for_strategy_development_ab" if passed else "reject_no_further_search"
        summary = {
            "asset_role": "research_only_l4_candidate_evaluation",
            "approval_status": "research_only_not_for_l5",
            "candidate_id": CANDIDATE_ID,
            "development_closed_after": base.DEVELOPMENT_CLOSED_AFTER,
            "sealed_periods_not_read": ["2025", "2026+"],
            "fold_results": results,
            "aggregate_delta_candidate_minus_baseline": aggregate,
            "model_layer_acceptance_gate": gate | {"passed": passed},
            "strategy_execution_acceptance_not_evaluated": True,
            "decision": decision,
            "ready_for_audit_review": True,
            "allow_next_layer_continue": False,
            "production_unchanged": True,
        }
        dump_json(output_dir / "evaluation_summary.json", summary)
        manifest = {
            "asset_role": "l4_research_prediction_asset",
            "approval_status": "research_only_not_for_l5",
            "candidate_id": CANDIDATE_ID,
            "method": "date-grouped XGBRanker rank:ndcg@10 additive residual on fold-local production-spec base margin",
            "frozen_contract_sha256": CONTRACT_SHA256,
            "oof_asset": str(oof_path), "oof_sha256": sha256_file(oof_path), "oof_quality": quality,
            "folds": [fold.__dict__ for fold in base.FOLDS],
            "decision": decision,
            "allow_for_strategy_development_ab_only": passed,
            "approved_for_l5": False,
            "allow_next_layer_continue": False,
            "production_unchanged": True,
        }
        dump_json(output_dir / "research_candidate_manifest.json", manifest)
        write_markdown(output_dir / "research_build_report.md", summary)
        handoff = {
            "status": "ready_for_audit_review", "candidate_id": CANDIDATE_ID, "decision": decision,
            "ready_for_audit_review": True, "allow_next_layer_continue": False,
            "allow_strategy_observation": passed, "allow_validation": False, "allow_production_change": False,
            "evidence": ["preflight.json", "evaluation_summary.json", "research_candidate_manifest.json", "baseline_candidate_same_key_oof.parquet"],
        }
        dump_json(output_dir / "audit_handoff.json", handoff)
        dump_json(output_dir / "hash_inventory.json", {
            "contract_sha256": CONTRACT_SHA256,
            "preflight_sha256": sha256_file(output_dir / "preflight.json"),
            "evaluation_summary_sha256": sha256_file(output_dir / "evaluation_summary.json"),
            "candidate_manifest_sha256": sha256_file(output_dir / "research_candidate_manifest.json"),
            "oof_sha256": sha256_file(oof_path),
            "audit_handoff_sha256": sha256_file(output_dir / "audit_handoff.json"),
            "models": model_hashes,
            "script_sha256": sha256_file(Path(__file__)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        return 0
    finally:
        feature_connection.close()
        label_connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
