"""Build a rolling OOF continuous-return reranker over production Top10."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from research_v260_pit_oof_excess_logit_10d_v1_20260816 import (
    canonical_frame_hash,
    json_dump,
    sha256_bytes,
    sha256_file,
)
from research_v260_cost_aware_top10_pairwise_residual_10d_v1_20260817 import (
    KEYS,
    score_metrics,
    strict_order,
)

CONTRACT = Path(
    "quant/data_file/runtime/agent_workspaces/strategy-agent/work/"
    "v260_top10_continuous_ltr_reranker_20260817/training_contract.json"
)
SOURCE = Path(
    "quant/data_file/reports/model_agent_v260_bounded_residual_rank_projection_10d_v1_20260816_r1/"
    "baseline_candidate_same_key_oof.parquet"
)
OUT = Path("quant/data_file/reports/model_agent_v260_top10_continuous_ltr_reranker_10d_v1_20260817_r1")
FOLDS = ["fold01", "fold02", "fold03"]


def top10(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = strict_order(frame, "baseline_oof_score_current")
    return ordered.loc[ordered["baseline_rank"] <= 10].copy()


def top10_in_model_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact row order consumed and later decoded by the reranker."""
    return top10(frame).sort_values(["trade_date", "baseline_rank"], kind="mergesort")


def apply_rerank(frame: pd.DataFrame, full_margin: np.ndarray | None) -> pd.DataFrame:
    ordered = strict_order(frame, "baseline_oof_score_current")
    pieces: list[pd.DataFrame] = []
    cursor = 0
    for _, group in ordered.groupby("trade_date", sort=True):
        group = group.sort_values("baseline_rank", kind="mergesort").copy()
        current = group.head(10).copy()
        if full_margin is None:
            current["rerank_margin"] = current["baseline_oof_score_current"].to_numpy(dtype="float64")
        else:
            current["rerank_margin"] = full_margin[cursor : cursor + len(current)]
            cursor += len(current)
        reranked = current.sort_values(
            ["rerank_margin", "stock_code"], ascending=[False, True], kind="mergesort"
        )["stock_code"].astype(str).tolist()
        original = current["stock_code"].astype(str).tolist()
        final_codes = reranked + group.iloc[10:]["stock_code"].astype(str).tolist()
        final_rank = {code: index + 1 for index, code in enumerate(final_codes)}
        group["final_rank"] = group["stock_code"].astype(str).map(final_rank).astype("int32")
        group["candidate_raw_score"] = (len(group) - group["final_rank"]).astype("float64")
        group["rank_changed"] = (group["final_rank"] != group["baseline_rank"]).astype("int8")
        group["top1_changed"] = int(reranked[0] != original[0])
        if set(reranked) != set(original):
            raise RuntimeError("top10_membership_changed")
        if group.loc[group["baseline_rank"] > 10, "rank_changed"].any():
            raise RuntimeError("outside_top10_changed")
        pieces.append(group)
    if full_margin is not None and cursor != len(full_margin):
        raise RuntimeError("prediction_row_count_mismatch")
    return pd.concat(pieces, ignore_index=True)


def build_replay(
    replay: int,
    source: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_root = root / "replays" / f"replay{replay}"
    replay_root.mkdir(parents=True, exist_ok=False)
    emitted: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    model_hashes: list[str] = []
    for index, fold_id in enumerate(FOLDS):
        test = source.loc[source["fold_id"] == fold_id].copy()
        if index == 0:
            projected = apply_rerank(test, None)
            model_hash = "IDENTITY_WARMUP_FOLD01"
            train_rows = 0
        else:
            train = top10_in_model_order(source.loc[source["fold_id"].isin(FOLDS[:index])].copy())
            test_top = top10_in_model_order(test)
            qid = pd.factorize(train["trade_date"], sort=True)[0].astype("int32")
            model = xgb.XGBRanker(**params)
            model.fit(
                train[features].astype("float32"),
                train["target"].astype("float32"),
                qid=qid,
                base_margin=train["baseline_oof_score_current"].astype("float32"),
                verbose=False,
            )
            full_margin = model.predict(
                test_top[features].astype("float32"),
                output_margin=True,
                base_margin=test_top["baseline_oof_score_current"].astype("float32"),
            ).astype("float64")
            projected = apply_rerank(test, full_margin)
            model_hash = sha256_bytes(model.get_booster().save_raw(raw_format="json"))
            train_rows = len(train)
        projected["fold_id"] = fold_id
        metrics = score_metrics(projected)
        metrics.update({"fold_id": fold_id, "train_rows": train_rows, "model_sha256": model_hash})
        fold_metrics.append(metrics)
        emitted.append(projected)
        model_hashes.append(model_hash)
        (replay_root / f"{fold_id}_model.sha256").write_text(model_hash + "\n", encoding="ascii")
    output = pd.concat(emitted, ignore_index=True).sort_values(KEYS, kind="mergesort")
    aggregate = score_metrics(output)
    active = fold_metrics[1:]
    gates = {
        "same_key": bool(len(output) == len(source) and not output.duplicated(KEYS).any()),
        "finite": bool(np.isfinite(output[["target", "candidate_raw_score"]]).all().all()),
        "top10_turnover_identical": bool(
            abs(aggregate["candidate_turnover_proxy"] - aggregate["baseline_turnover_proxy"]) <= 1e-15
            and all(abs(item["candidate_turnover_proxy"] - item["baseline_turnover_proxy"]) <= 1e-15 for item in fold_metrics)
        ),
        "top1_not_worse": bool(
            aggregate["candidate_top1"] >= aggregate["baseline_top1"]
            and all(item["candidate_top1"] >= item["baseline_top1"] for item in active)
        ),
        "top3_not_worse": bool(
            aggregate["candidate_top3"] >= aggregate["baseline_top3"]
            and all(item["candidate_top3"] >= item["baseline_top3"] for item in active)
        ),
        "effective_change": bool(all(item["top1_changed"] > 0 for item in active)),
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
    root = Path(args.output_dir)
    if root.exists():
        raise RuntimeError(f"refusing_existing_output: {root}")
    root.mkdir(parents=True)
    try:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        if contract["candidate_count"] != 1 or xgb.__version__ != "2.1.4":
            raise RuntimeError("contract_identity_or_xgboost_version")
        source = pd.read_parquet(SOURCE).rename(columns={"executable_10d_open_return": "target"})
        features = [
            name
            for name in source.columns
            if name
            not in {
                "trade_date", "stock_code", "target", "baseline_oof_score_current",
                "baseline_oof_rank_current", "baseline_oof_rank_pct_current",
                "executable_10d_net_excess_event", "raw_residual_margin", "clipped_residual",
                "preference_score", "final_rank", "candidate_raw_score", "rank_changed", "fold_id",
            }
        ]
        if len(features) != 40 or source["trade_date"].max() > "20241231":
            raise RuntimeError("feature_or_development_boundary")
        json_dump(
            root / "preflight.json",
            {
                "candidate_id": contract["candidate_id"],
                "contract_sha256": sha256_file(CONTRACT),
                "source_sha256": sha256_file(SOURCE),
                "rows": len(source),
                "features": features,
                "development_only": True,
                "production_unchanged": True,
            },
        )
        params = dict(contract["model"]["params"])
        replays = [build_replay(index, source, features, params, root) for index in (1, 2, 3)]
        output, first = replays[0]
        deterministic = all(item[1]["hashes"] == first["hashes"] for item in replays[1:])
        gates = dict(first["gates"])
        gates["deterministic_3_of_3"] = deterministic
        decision = "model_layer_passed_ready_for_strategy_ab" if all(gates.values()) else "reject_no_further_search"
        output.to_parquet(root / "baseline_candidate_same_key_oof.parquet", index=False)
        json_dump(
            root / "evaluation_summary.json",
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
        json_dump(root / "build_failure.json", {"status": "build_failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
