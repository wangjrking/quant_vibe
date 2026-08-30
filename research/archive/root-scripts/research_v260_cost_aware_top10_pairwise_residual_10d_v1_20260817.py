"""Build one cost-aware, production-anchored Top10 pairwise research score."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from research_v260_pit_oof_excess_logit_10d_v1_20260816 import (
    DEVELOPMENT_END,
    FEATURE_DB,
    FEATURE_TABLE,
    METADATA_PATH,
    canonical_frame_hash,
    feature_aliases,
    json_dump,
    label_end_map,
    load_calendar,
    load_data,
    production_params,
    quote,
    sha256_bytes,
    sha256_file,
    strict_monthly_baseline_state,
)

CONTRACT = Path(
    "quant/data_file/runtime/agent_workspaces/strategy-agent/work/"
    "v260_cost_aware_top10_pairwise_residual_20260817/training_contract.json"
)
BASELINE_OOF = Path(
    "quant/data_file/reports/model_agent_v260_bounded_residual_rank_projection_10d_v1_20260816_r1/"
    "baseline_candidate_same_key_oof.parquet"
)
OUT = Path("quant/data_file/reports/model_agent_v260_cost_aware_top10_pairwise_residual_10d_v1_20260817_r1")
START = "20220104"
KEYS = ["trade_date", "stock_code"]


def strict_order(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    out = frame.sort_values(
        ["trade_date", score, "stock_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    out["baseline_rank"] = out.groupby("trade_date", sort=False).cumcount() + 1
    return out


def make_adjacent_pairs(
    frame: pd.DataFrame,
    features: list[str],
    hurdle: float,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, int]]:
    ordered = strict_order(frame, "baseline_state_score")
    ordered = ordered.loc[ordered["baseline_rank"] <= top_n].copy()
    pair_rows: list[dict[str, Any]] = []
    ambiguous = 0
    raw_pairs = 0
    for trade_date, group in ordered.groupby("trade_date", sort=True):
        group = group.sort_values("baseline_rank", kind="mergesort").reset_index(drop=True)
        for index in range(len(group) - 1):
            upper = group.iloc[index]
            lower = group.iloc[index + 1]
            if not np.isfinite([upper["target"], lower["target"]]).all():
                continue
            raw_pairs += 1
            gap = float(lower["target"] - upper["target"])
            if abs(gap) <= hurdle:
                ambiguous += 1
                continue
            diff = {
                f"d_{name}": float(lower[name]) - float(upper[name])
                if np.isfinite([lower[name], upper[name]]).all()
                else np.nan
                for name in features
            }
            diff.update(
                {
                    "trade_date": str(trade_date),
                    "upper_code": str(upper["stock_code"]),
                    "lower_code": str(lower["stock_code"]),
                    "production_score_gap": float(lower["baseline_state_score"] - upper["baseline_state_score"]),
                    "label": int(gap > hurdle),
                }
            )
            pair_rows.append(diff)
            reverse = {
                key: (-value if key.startswith("d_") or key == "production_score_gap" else value)
                for key, value in diff.items()
            }
            reverse["upper_code"], reverse["lower_code"] = diff["lower_code"], diff["upper_code"]
            reverse["label"] = 1 - diff["label"]
            pair_rows.append(reverse)
    if not pair_rows:
        raise RuntimeError("no_cost_clear_training_pairs")
    return pd.DataFrame(pair_rows), {
        "raw_adjacent_pairs": raw_pairs,
        "ambiguous_pairs_dropped": ambiguous,
        "symmetrized_training_rows": len(pair_rows),
    }


def _pair_vector(
    lookup: dict[str, np.ndarray],
    score_lookup: dict[str, float],
    left: str,
    right: str,
) -> np.ndarray:
    return np.concatenate(
        [lookup[right] - lookup[left], np.asarray([score_lookup[right] - score_lookup[left]], dtype="float32")]
    )


def project_top10(
    frame: pd.DataFrame,
    features: list[str],
    predict_probability: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    phases: int = 2,
) -> pd.DataFrame:
    ordered = strict_order(frame, "baseline_oof_score_current")
    pieces: list[pd.DataFrame] = []
    for _, group in ordered.groupby("trade_date", sort=True):
        group = group.sort_values("baseline_rank", kind="mergesort").copy()
        codes = group["stock_code"].astype(str).tolist()
        work = codes[:10]
        lookup = {
            str(row.stock_code): row[features].to_numpy(dtype="float32")
            for _, row in group.head(10).iterrows()
        }
        score_lookup = dict(
            zip(group.head(10)["stock_code"].astype(str), group.head(10)["baseline_oof_score_current"].astype(float))
        )
        for phase in range(phases):
            positions = list(range(phase % 2, len(work) - 1, 2))
            if not positions:
                continue
            matrix = np.vstack(
                [_pair_vector(lookup, score_lookup, work[pos], work[pos + 1]) for pos in positions]
            )
            probabilities = np.asarray(predict_probability(matrix), dtype="float64")
            if probabilities.shape != (len(positions),) or not np.isfinite(probabilities).all():
                raise RuntimeError("invalid_pair_probability")
            for pos, probability in zip(positions, probabilities):
                if probability >= threshold:
                    work[pos], work[pos + 1] = work[pos + 1], work[pos]
        final_codes = work + codes[10:]
        final_rank = {code: index + 1 for index, code in enumerate(final_codes)}
        group["final_rank"] = group["stock_code"].astype(str).map(final_rank).astype("int32")
        group["candidate_raw_score"] = (len(group) - group["final_rank"]).astype("float64")
        group["rank_changed"] = (group["final_rank"] != group["baseline_rank"]).astype("int8")
        group["top1_changed"] = int(final_codes[0] != codes[0])
        if set(final_codes[:10]) != set(codes[:10]):
            raise RuntimeError("top10_membership_changed")
        if (group["final_rank"] - group["baseline_rank"]).abs().max() > phases:
            raise RuntimeError("rank_displacement_exceeded")
        if group.loc[group["baseline_rank"] > 10, "rank_changed"].any():
            raise RuntimeError("outside_top10_order_changed")
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def turnover_proxy(frame: pd.DataFrame, score: str) -> float:
    sets: list[set[str]] = []
    for _, group in frame.groupby("trade_date", sort=True):
        top = group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort").head(10)
        sets.append(set(top["stock_code"].astype(str)))
    if len(sets) < 2:
        raise RuntimeError("insufficient_turnover_dates")
    return float(np.mean([1 - len(sets[index] & sets[index - 1]) / 10 for index in range(1, len(sets))]))


def score_metrics(frame: pd.DataFrame) -> dict[str, float]:
    daily: list[dict[str, float]] = []
    for _, group in frame.groupby("trade_date", sort=True):
        baseline = group.sort_values(
            ["baseline_oof_score_current", "stock_code"], ascending=[False, True], kind="mergesort"
        )
        candidate = group.sort_values(
            ["candidate_raw_score", "stock_code"], ascending=[False, True], kind="mergesort"
        )
        daily.append(
            {
                "baseline_top1": float(baseline.head(1)["target"].mean()),
                "candidate_top1": float(candidate.head(1)["target"].mean()),
                "baseline_top3": float(baseline.head(3)["target"].mean()),
                "candidate_top3": float(candidate.head(3)["target"].mean()),
                "rank_corr": float(group["baseline_rank"].corr(group["final_rank"], method="spearman")),
                "top1_changed": float(group["top1_changed"].iloc[0]),
                "changed_top10_row_share": float(group.loc[group["baseline_rank"] <= 10, "rank_changed"].mean()),
            }
        )
    d = pd.DataFrame(daily)
    return {
        **{name: float(d[name].mean()) for name in d.columns},
        "baseline_turnover_proxy": turnover_proxy(frame, "baseline_oof_score_current"),
        "candidate_turnover_proxy": turnover_proxy(frame, "candidate_raw_score"),
        "metric_days": int(len(d)),
    }


def fit_and_project(
    replay: int,
    train_source: pd.DataFrame,
    test_oof: pd.DataFrame,
    features: list[str],
    end_map: dict[str, str],
    folds: list[dict[str, str]],
    contract: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_root = output_root / "replays" / f"replay{replay}"
    replay_root.mkdir(parents=True, exist_ok=False)
    rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    model_hashes: list[str] = []
    hurdle = float(contract["training_target"]["round_trip_hurdle"])
    threshold = float(contract["output_projection"]["swap_probability_threshold"])
    pair_columns = [f"d_{name}" for name in features] + ["production_score_gap"]
    for fold in folds:
        train = train_source.loc[
            (train_source["trade_date"] >= START)
            & (train_source["trade_date"] < fold["test_start"])
            & (train_source["trade_date"].map(end_map) < fold["test_start"])
            & np.isfinite(train_source["target"])
        ].copy()
        test = test_oof.loc[test_oof["fold_id"] == fold["fold_id"]].copy()
        pairs, pair_summary = make_adjacent_pairs(train, features, hurdle)
        params = dict(contract["model"]["params"])
        model = xgb.XGBClassifier(**params)
        model.fit(pairs[pair_columns].astype("float32"), pairs["label"].astype("int8"), verbose=False)

        def predict(matrix: np.ndarray) -> np.ndarray:
            return model.predict_proba(matrix)[:, 1]

        projected = project_top10(test, features, predict, threshold, int(contract["output_projection"]["phases"]))
        projected["fold_id"] = fold["fold_id"]
        metrics = score_metrics(projected)
        metrics.update({"fold_id": fold["fold_id"], "train_rows": len(train), **pair_summary})
        fold_metrics.append(metrics)
        rows.append(projected)
        model_hash = sha256_bytes(model.get_booster().save_raw(raw_format="json"))
        model_hashes.append(model_hash)
        (replay_root / f"{fold['fold_id']}_model.sha256").write_text(model_hash + "\n", encoding="ascii")
    output = pd.concat(rows, ignore_index=True).sort_values(KEYS, kind="mergesort")
    aggregate = score_metrics(output)
    checks = [aggregate, *fold_metrics]
    gates = {
        "same_key": bool(not output.duplicated(KEYS).any() and len(output) == len(test_oof)),
        "finite": bool(np.isfinite(output[["baseline_oof_score_current", "candidate_raw_score", "target"]]).all().all()),
        "top10_membership_exact": all(
            abs(item["candidate_turnover_proxy"] - item["baseline_turnover_proxy"]) <= 1e-15 for item in checks
        ),
        "top1_not_worse": all(item["candidate_top1"] >= item["baseline_top1"] for item in checks),
        "top3_not_worse": all(item["candidate_top3"] >= item["baseline_top3"] for item in checks),
        "sparse_top1_change": all(0 < item["top1_changed"] <= 0.25 for item in checks),
        "sparse_top10_change": all(0 < item["changed_top10_row_share"] <= 0.20 for item in checks),
    }
    hashes = {
        "baseline": canonical_frame_hash(output, KEYS + ["baseline_oof_score_current"]),
        "candidate": canonical_frame_hash(output, KEYS + ["candidate_raw_score"]),
        "models": sha256_bytes("".join(model_hashes).encode("ascii")),
    }
    summary = {"replay": replay, "aggregate": aggregate, "fold_metrics": fold_metrics, "gates": gates, "hashes": hashes}
    json_dump(replay_root / "replay_summary.json", summary)
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    if output_root.exists():
        raise RuntimeError(f"refusing_existing_output: {output_root}")
    output_root.mkdir(parents=True)
    try:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        if contract["candidate_count"] != 1 or xgb.__version__ != contract["model"]["xgboost_version"]:
            raise RuntimeError("contract_identity_or_xgboost_version")
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        connection = duckdb.connect(str(FEATURE_DB), read_only=True)
        try:
            available = {row[1] for row in connection.execute(f"pragma table_info({quote(FEATURE_TABLE)})").fetchall()}
        finally:
            connection.close()
        requested, features, aliases = feature_aliases(metadata, available)
        if len(features) != 40:
            raise RuntimeError("production_feature_count_changed")
        calendar = load_calendar()
        end_map = label_end_map(calendar)
        data = load_data(features)
        if data["trade_date"].max() > DEVELOPMENT_END:
            raise RuntimeError("development_boundary_violation")
        state, monthly_hashes = strict_monthly_baseline_state(
            data,
            features,
            calendar,
            end_map,
            production_params(metadata),
            output_root / "monthly_baseline_models",
        )
        train_source = data.merge(state[KEYS + ["baseline_state_score"]], on=KEYS, how="inner", validate="one_to_one")
        test_oof = pd.read_parquet(BASELINE_OOF)
        test_oof = test_oof.rename(columns={"executable_10d_open_return": "target"})
        folds = [
            {"fold_id": "fold01", "test_start": "20220620", "test_end": "20221230"},
            {"fold_id": "fold02", "test_start": "20230130", "test_end": "20231229"},
            {"fold_id": "fold03", "test_start": "20240129", "test_end": "20241213"},
        ]
        json_dump(
            output_root / "preflight.json",
            {
                "candidate_id": contract["candidate_id"],
                "contract_sha256": sha256_file(CONTRACT),
                "baseline_oof_sha256": sha256_file(BASELINE_OOF),
                "features": requested,
                "aliases": aliases,
                "development_only": True,
                "production_unchanged": True,
                "monthly_state_models": len(monthly_hashes),
            },
        )
        replays = [fit_and_project(index, train_source, test_oof, features, end_map, folds, contract, output_root) for index in (1, 2, 3)]
        output, first = replays[0]
        deterministic = all(item[1]["hashes"] == first["hashes"] for item in replays[1:])
        gates = dict(first["gates"])
        gates["deterministic_3_of_3"] = deterministic
        decision = "model_layer_passed_ready_for_strategy_ab" if all(gates.values()) else "reject_no_further_search"
        output.to_parquet(output_root / "baseline_candidate_same_key_oof.parquet", index=False)
        json_dump(
            output_root / "evaluation_summary.json",
            {
                "candidate_id": contract["candidate_id"],
                "decision": decision,
                "aggregate": first["aggregate"],
                "fold_metrics": first["fold_metrics"],
                "gates": gates,
                "deterministic_hashes": [item[1]["hashes"] for item in replays],
                "same_key_rows": len(output),
                "sealed_2025_2026_not_read": True,
                "production_unchanged": True,
            },
        )
        return 0
    except Exception as exc:
        json_dump(
            output_root / "build_failure.json",
            {
                "status": "build_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "production_unchanged": True,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
