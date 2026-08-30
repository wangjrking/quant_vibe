"""Evaluate the one frozen Top20-local HGB rank projection research candidate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash, evaluate, sha256_file
from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics, dump_json


CANDIDATE_ID = "v261_10d_hist_top20_rank_projection_v1"
CONTRACT = Path("quant/data_file/reports/model_agent_10d_hist_top20_rank_projection_20260830/training_contract.json")
HGB_OOF = Path("quant/data_file/reports/model_agent_10d_hist_gradient_structure_20260830/build_r1/baseline_candidate_same_key_oof.parquet")
DEFAULT_OUTPUT = Path("quant/data_file/reports/model_agent_10d_hist_top20_rank_projection_20260830/build_r1")
PREFIX = 20


def project_one_day(group: pd.DataFrame) -> pd.DataFrame:
    production = group.sort_values(
        ["baseline_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort"
    ).copy()
    prefix = production.head(PREFIX).sort_values(
        ["candidate_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort"
    )
    tail = production.iloc[PREFIX:]
    projected = pd.concat([prefix, tail], ignore_index=True)
    projected["baseline_rank"] = projected["stock_code"].map(
        {code: position + 1 for position, code in enumerate(production["stock_code"])}
    ).astype("int32")
    projected["final_rank"] = np.arange(1, len(projected) + 1, dtype="int32")
    projected["candidate_raw_score"] = (len(projected) - projected["final_rank"]).astype("float64")
    if not np.array_equal(projected.iloc[PREFIX:]["stock_code"].to_numpy(), tail["stock_code"].to_numpy()):
        raise RuntimeError("blocked_tail_order_changed")
    return projected


def project(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [project_one_day(group) for _, group in frame.groupby("trade_date", sort=True)],
        ignore_index=True,
    ).sort_values(["trade_date", "stock_code"], kind="mergesort").reset_index(drop=True)


def score_hash(frame: pd.DataFrame) -> str:
    return canonical_frame_hash(frame, ["trade_date", "stock_code", "candidate_raw_score"])


def run(output: Path) -> int:
    if output.exists():
        raise RuntimeError("blocked_existing_output")
    output.mkdir(parents=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["candidate_id"] != CANDIDATE_ID or contract["algorithm"]["mutable_prefix_size"] != PREFIX:
        raise RuntimeError("blocked_contract_mismatch")
    source = pd.read_parquet(HGB_OOF)
    required = {"trade_date", "stock_code", "target", "baseline_pred_prob", "candidate_pred_prob", "fold_id"}
    if not required.issubset(source.columns):
        raise RuntimeError("blocked_hgb_oof_schema")
    if source["trade_date"].astype(str).str[:4].isin(["2025", "2026"]).any():
        raise RuntimeError("blocked_sealed_window_read")
    if source.duplicated(["trade_date", "stock_code"]).any() or source.stock_code.astype(str).str.endswith(".BJ").any():
        raise RuntimeError("blocked_source_key_or_bj")
    if not np.isfinite(source[["target", "baseline_pred_prob", "candidate_pred_prob"]].to_numpy(dtype="float64")).all():
        raise RuntimeError("blocked_source_nonfinite")
    dump_json(output / "preflight.json", {
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": sha256_file(CONTRACT),
        "hgb_oof_sha256": sha256_file(HGB_OOF),
        "development_window": ["20220101", "20241231"],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "algorithm": "production_top20_reordered_once_by_hgb; all_ranks_after_20_unchanged",
        "production_unchanged": True,
    })
    replays = [project(source) for _ in range(3)]
    replay_hashes = [score_hash(frame) for frame in replays]
    if len(set(replay_hashes)) != 1:
        raise RuntimeError("blocked_deterministic_replay")
    oof = replays[0]
    base = oof.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]]
    candidate = oof.rename(columns={"candidate_raw_score": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]]
    base_metrics, _ = evaluate(base)
    candidate_metrics, _ = evaluate(candidate)
    folds: dict[str, object] = {}
    for fold_id, group in oof.groupby("fold_id", sort=True):
        fold_base, _ = evaluate(group.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
        fold_candidate, _ = evaluate(group.rename(columns={"candidate_raw_score": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
        folds[str(fold_id)] = {
            "rows": int(len(group)),
            "baseline_metrics": fold_base,
            "candidate_metrics": fold_candidate,
            "failed_gates": compare_metrics(fold_base, fold_candidate),
        }
    failures = compare_metrics(base_metrics, candidate_metrics)
    failures += [failure for item in folds.values() for failure in item["failed_gates"]]
    changed_share = float((oof["baseline_rank"] != oof["final_rank"]).mean())
    if changed_share == 0:
        failures.append("nonzero_effective_change")
    failures = sorted(set(failures))
    passed = not failures
    oof.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "status": "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search",
        "same_key_rows": int(len(oof)),
        "same_key_duplicate_groups": int(oof.duplicated(["trade_date", "stock_code"]).sum()),
        "bj_rows": int(oof.stock_code.astype(str).str.endswith(".BJ").sum()),
        "null_or_nonfinite_rows": int((~np.isfinite(oof[["target", "baseline_pred_prob", "candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
        "projection": {"mutable_prefix_size": PREFIX, "changed_rank_row_share": changed_share},
        "aggregate": {"baseline_metrics": base_metrics, "candidate_metrics": candidate_metrics, "failed_gates": compare_metrics(base_metrics, candidate_metrics)},
        "folds": folds,
        "hard_gate_passed": passed,
        "failed_gates": failures,
        "deterministic": {"passed": True, "replay_candidate_score_sha256": replay_hashes},
        "baseline_score_sha256": canonical_frame_hash(base, ["trade_date", "stock_code", "pred_prob"]),
        "candidate_score_sha256": canonical_frame_hash(candidate, ["trade_date", "stock_code", "pred_prob"]),
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    }
    dump_json(output / "evaluation_summary.json", summary)
    dump_json(output / "research_candidate_manifest.json", {
        "candidate_id": CANDIDATE_ID,
        "approval_status": "research_only_not_for_l5",
        "decision": summary["status"],
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    })
    dump_json(output / "hash_inventory.json", {
        "script_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(CONTRACT),
        "source_hgb_oof_sha256": sha256_file(HGB_OOF),
        "oof_sha256": sha256_file(output / "baseline_candidate_same_key_oof.parquet"),
        "summary_sha256": sha256_file(output / "evaluation_summary.json"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(run(DEFAULT_OUTPUT))
