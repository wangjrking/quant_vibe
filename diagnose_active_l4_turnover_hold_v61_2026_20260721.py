# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_turnover_hold_v61_20260721 as v61


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant/data_file/reports/strategy_agent_active_l4_turnover_hold_v61_20260721"
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_turnover_hold_v61_2026_diagnostic_20260721"
PROTOCOL = OUT / "diagnostic_protocol.json"


def slice_by_date(arrays, state, start_date, end_date):
    dates = arrays["dates"].astype(str)
    indices = np.flatnonzero((dates >= start_date) & (dates <= end_date))
    if len(indices) < 2 or np.any(np.diff(indices) != 1):
        raise RuntimeError("诊断窗口日期不连续或不足两个交易日")
    start, stop = int(indices[0]), int(indices[-1]) + 1
    sliced_arrays = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(dates):
            sliced_arrays[key] = value[start:stop]
        else:
            sliced_arrays[key] = value
    sliced_state = tuple(
        value[start:stop] if isinstance(value, np.ndarray) and value.shape[0] == len(dates) else value
        for value in state
    )
    return sliced_arrays, sliced_state


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_posthoc_diagnostic":
        raise RuntimeError("诊断协议未冻结")
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("代码哈希不一致")

    source_protocol_path = ROOT / protocol["source_protocol"]["path"]
    candidates_path = ROOT / protocol["source_candidates"]["path"]
    cache_path = ROOT / protocol["input_cache"]["path"]
    for path, expected in (
        (source_protocol_path, protocol["source_protocol"]["sha256"]),
        (candidates_path, protocol["source_candidates"]["sha256"]),
        (cache_path, protocol["input_cache"]["sha256"]),
    ):
        if metrics_lib.digest(path) != expected:
            raise RuntimeError(f"输入哈希不一致: {path}")

    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    document = json.loads(candidates_path.read_text(encoding="utf-8"))
    matches = [row for row in document["profiles"] if row["case_id"] == protocol["candidate_id"]]
    if len(matches) != 1:
        raise RuntimeError("冻结候选不存在或不唯一")
    profile = matches[0]

    with np.load(cache_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    fixed = dict(source_protocol["fixed"])
    fixed["exposure_floor"] = float(profile["exposure_floor"])
    state = robust.build_state(
        arrays,
        source_protocol["fixed"]["score_weights"],
        json.loads(profile["agreement_json"]),
        float(json.loads(profile["entry_json"])["risk_off"]),
        float(json.loads(profile["entry_json"])["risk_on"]),
        fixed,
    )
    state = v61.apply_turnover_filter(state, arrays, float(profile["turnover_max"]))
    validation_arrays, validation_state = slice_by_date(
        arrays,
        state,
        protocol["diagnostic_start"],
        protocol["diagnostic_end"],
    )
    local_protocol = dict(source_protocol)
    local_protocol["observation_end"] = protocol["diagnostic_end"]
    exit_profile = sell_lib.ExitProfile(
        profile["exit_score_source"],
        int(profile["min_hold"]),
        int(profile["max_hold"]),
        float(profile["sell_rank_below"]),
        float(profile["replacement_advantage"]),
    )
    daily, actions = v61.run_case(
        validation_arrays,
        local_protocol,
        validation_state,
        int(profile["top_n"]),
        exit_profile,
        record_actions=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    action_path = OUT / f"{protocol['candidate_id']}_2026_diagnostic.csv"
    daily_path = OUT / f"{protocol['candidate_id']}_2026_local_daily.csv"
    actions.to_csv(action_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    result = {
        "status": "posthoc_diagnostic_actions_pending_juejin",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_id": protocol["candidate_id"],
        "diagnostic_start": protocol["diagnostic_start"],
        "diagnostic_end": protocol["diagnostic_end"],
        "empty_start": True,
        "unseen_validation_claim_allowed": False,
        "action_path": str(action_path.relative_to(ROOT)).replace("\\", "/"),
        "action_sha256": metrics_lib.digest(action_path),
        "action_rows": len(actions),
        "buy_rows": int((actions.action == "BUY").sum()) if len(actions) else 0,
        "sell_rows": int((actions.action == "SELL").sum()) if len(actions) else 0,
        "daily_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
        "production_changed": False,
    }
    (OUT / "diagnostic_action_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
