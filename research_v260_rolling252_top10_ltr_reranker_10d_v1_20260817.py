"""Build one monthly rolling 252-date Top10 learning-to-rank candidate."""
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
    label_end_map,
    load_calendar,
    sha256_bytes,
    sha256_file,
)
from research_v260_cost_aware_top10_pairwise_residual_10d_v1_20260817 import KEYS, score_metrics, strict_order
from research_v260_top10_continuous_ltr_reranker_10d_v1_20260817 import (
    FOLDS,
    SOURCE,
    apply_rerank,
    top10_in_model_order,
)

CONTRACT = Path(
    "quant/data_file/runtime/agent_workspaces/strategy-agent/work/"
    "v260_rolling252_top10_ltr_reranker_20260817/training_contract.json"
)
OUT = Path("quant/data_file/reports/model_agent_v260_rolling252_top10_ltr_reranker_10d_v1_20260817_r1")
WINDOW = 252


def admitted_dates(source: pd.DataFrame, end_map: dict[str, str], prediction_start: str) -> list[str]:
    dates = sorted(source["trade_date"].astype(str).unique())
    return [date for date in dates if date < prediction_start and end_map.get(date, "99999999") < prediction_start]


def build_replay(
    replay: int,
    source: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    end_map: dict[str, str],
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_root = root / "replays" / f"replay{replay}"
    replay_root.mkdir(parents=True, exist_ok=False)
    monthly_outputs: list[pd.DataFrame] = []
    monthly_models: list[dict[str, Any]] = []
    source = source.copy()
    source["month"] = source["trade_date"].str[:6]
    for month in sorted(source["month"].unique()):
        test = source.loc[source["month"] == month].drop(columns="month").copy()
        prediction_start = str(test["trade_date"].min())
        dates = admitted_dates(source, end_map, prediction_start)
        if len(dates) < WINDOW:
            test["reranker_margin"] = test["baseline_oof_score_current"].astype("float64")
            test["reranker_active"] = np.int8(0)
            projected = apply_rerank(test, None)
            model_hash = "IDENTITY_COLD_START"
            train_date_count = len(dates)
        else:
            selected_dates = dates[-WINDOW:]
            train = top10_in_model_order(source.loc[source["trade_date"].isin(selected_dates)].drop(columns="month").copy())
            test_all = strict_order(test, "baseline_oof_score_current").sort_values(
                ["trade_date", "baseline_rank"], kind="mergesort"
            )
            qid = pd.factorize(train["trade_date"], sort=True)[0].astype("int32")
            model = xgb.XGBRanker(**params)
            model.fit(
                train[features].astype("float32"),
                train["target"].astype("float32"),
                qid=qid,
                base_margin=train["baseline_oof_score_current"].astype("float32"),
                verbose=False,
            )
            margins = model.predict(
                test_all[features].astype("float32"),
                output_margin=True,
                base_margin=test_all["baseline_oof_score_current"].astype("float32"),
            ).astype("float64")
            margin_frame = test_all[KEYS].copy()
            margin_frame["reranker_margin"] = margins
            test = test.merge(margin_frame, on=KEYS, how="left", validate="one_to_one")
            test["reranker_active"] = np.int8(1)
            top10_margins = top10_in_model_order(test)["reranker_margin"].to_numpy(dtype="float64")
            projected = apply_rerank(test, top10_margins)
            model_hash = sha256_bytes(model.get_booster().save_raw(raw_format="json"))
            train_date_count = len(selected_dates)
        monthly_outputs.append(projected)
        monthly_models.append(
            {
                "month": str(month),
                "prediction_start": prediction_start,
                "train_date_count": train_date_count,
                "model_sha256": model_hash,
            }
        )
    output = pd.concat(monthly_outputs, ignore_index=True).sort_values(KEYS, kind="mergesort")
    aggregate = score_metrics(output)
    fold_metrics = []
    for fold_id in FOLDS:
        metrics = score_metrics(output.loc[output["fold_id"] == fold_id].copy())
        metrics["fold_id"] = fold_id
        fold_metrics.append(metrics)
    active_folds = [item for item in fold_metrics if item["top1_changed"] > 0]
    gates = {
        "same_key": bool(len(output) == len(source) and not output.duplicated(KEYS).any()),
        "finite": bool(np.isfinite(output[["target", "candidate_raw_score"]]).all().all()),
        "top10_turnover_identical": bool(
            abs(aggregate["candidate_turnover_proxy"] - aggregate["baseline_turnover_proxy"]) <= 1e-15
            and all(abs(item["candidate_turnover_proxy"] - item["baseline_turnover_proxy"]) <= 1e-15 for item in fold_metrics)
        ),
        "top1_not_worse": bool(
            aggregate["candidate_top1"] >= aggregate["baseline_top1"]
            and all(item["candidate_top1"] >= item["baseline_top1"] for item in fold_metrics)
        ),
        "top3_not_worse": bool(
            aggregate["candidate_top3"] >= aggregate["baseline_top3"]
            and all(item["candidate_top3"] >= item["baseline_top3"] for item in fold_metrics)
        ),
        "effective_change": bool(active_folds and all(item["top1_changed"] > 0 for item in active_folds)),
    }
    hashes = {
        "baseline": canonical_frame_hash(output, KEYS + ["baseline_oof_score_current"]),
        "candidate": canonical_frame_hash(output, KEYS + ["candidate_raw_score"]),
        "models": sha256_bytes("".join(item["model_sha256"] for item in monthly_models).encode("ascii")),
    }
    summary = {
        "replay": replay,
        "aggregate": aggregate,
        "fold_metrics": fold_metrics,
        "monthly_models": monthly_models,
        "gates": gates,
        "hashes": hashes,
    }
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
        end_map = label_end_map(load_calendar())
        json_dump(
            root / "preflight.json",
            {
                "candidate_id": contract["candidate_id"],
                "contract_sha256": sha256_file(CONTRACT),
                "source_sha256": sha256_file(SOURCE),
                "rows": len(source),
                "window_signal_dates": WINDOW,
                "development_only": True,
                "production_unchanged": True,
            },
        )
        params = dict(contract["model"]["params"])
        replays = [build_replay(index, source, features, params, end_map, root) for index in (1, 2, 3)]
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
