# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_lagged_health_liquidity_v66_20260721 as v66


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant/data_file/reports/strategy_agent_active_l4_lagged_health_liquidity_v66_20260721"
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_lagged_health_validation_v67_20260721"
PROTOCOL = OUT / "validation_protocol.json"


def slice_by_date(arrays, start_date, end_date):
    dates = arrays["dates"].astype(str)
    indices = np.flatnonzero((dates >= start_date) & (dates <= end_date))
    if len(indices) < 2 or np.any(np.diff(indices) != 1):
        raise RuntimeError("验证窗口日期不连续或不足两个交易日")
    start, stop = int(indices[0]), int(indices[-1]) + 1
    result = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(dates):
            result[key] = value[start:stop]
        else:
            result[key] = value
    return result, start, stop


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_validation_once":
        raise RuntimeError("验证协议未冻结")
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("验证代码哈希不一致")

    source_protocol_path = ROOT / protocol["source_protocol"]["path"]
    source_candidates_path = ROOT / protocol["source_candidates"]["path"]
    cache_path = ROOT / protocol["input_cache"]["path"]
    for path, expected in (
        (source_protocol_path, protocol["source_protocol"]["sha256"]),
        (source_candidates_path, protocol["source_candidates"]["sha256"]),
        (cache_path, protocol["input_cache"]["sha256"]),
    ):
        if metrics_lib.digest(path) != expected:
            raise RuntimeError(f"输入资产哈希不一致: {path}")

    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    candidate_document = json.loads(source_candidates_path.read_text(encoding="utf-8"))
    matches = [
        row for row in candidate_document["profiles"]
        if row["case_id"] == protocol["candidate_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError("冻结候选不存在或不唯一")
    profile = matches[0]
    definition = {
        "weights": json.loads(profile["weights_json"]),
        "agreement": json.loads(profile["agreement_json"]),
        "entry": json.loads(profile["entry_json"]),
        "shadow_top_n": int(profile["shadow_top_n"]),
        "lookback": int(profile["lookback"]),
        "threshold": float(profile["threshold"]),
        "weak_floor": float(profile["weak_floor"]),
        "exit": {
            "exit_score_source": profile["exit_score_source"],
            "min_hold": int(profile["min_hold"]),
            "max_hold": int(profile["max_hold"]),
            "sell_rank_below": float(profile["sell_rank_below"]),
            "replacement_advantage": float(profile["replacement_advantage"]),
        },
    }

    with np.load(cache_path, allow_pickle=False) as saved:
        full_arrays = {key: saved[key] for key in saved.files}
    full_state = v66.state_with_health(full_arrays, source_protocol, definition)
    validation_arrays, start, stop = slice_by_date(
        full_arrays,
        protocol["validation_start"],
        protocol["validation_end"],
    )
    validation_state = tuple(
        value[start:stop] if isinstance(value, np.ndarray) and value.shape[0] == len(full_arrays["dates"]) else value
        for value in full_state
    )
    validation_protocol = dict(source_protocol)
    validation_protocol["observation_end"] = protocol["validation_end"]
    exit_profile = sell_lib.ExitProfile(**definition["exit"])
    daily, actions = v66.run_case(
        validation_arrays,
        validation_protocol,
        validation_state,
        exit_profile,
        record_actions=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    action_path = OUT / f"{protocol['candidate_id']}_validation_2026.csv"
    daily_path = OUT / f"{protocol['candidate_id']}_validation_2026_local_daily.csv"
    actions.to_csv(action_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    result = {
        "status": "validation_actions_frozen_pending_juejin",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_id": protocol["candidate_id"],
        "validation_start": protocol["validation_start"],
        "validation_end": protocol["validation_end"],
        "empty_start": True,
        "observation_positions_inherited": False,
        "parameters_changed_after_validation_open": False,
        "action_path": str(action_path.relative_to(ROOT)).replace("\\", "/"),
        "action_sha256": metrics_lib.digest(action_path),
        "action_rows": len(actions),
        "buy_rows": int((actions.action == "BUY").sum()) if len(actions) else 0,
        "sell_rows": int((actions.action == "SELL").sum()) if len(actions) else 0,
        "daily_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
        "daily_sha256": metrics_lib.digest(daily_path),
        "definition": {**definition, "exit": asdict(exit_profile)},
        "production_changed": False,
    }
    (OUT / "validation_action_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
