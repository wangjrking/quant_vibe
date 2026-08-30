# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_fast_health_neighborhood_v75_20260722 as v75
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_lagged_health_liquidity_v66_20260721 as health_lib


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_fast_health_v76_validation_20260722"
PROTOCOL = OUT / "validation_protocol.json"


def slice_time(value, start_index, total_dates):
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total_dates:
        return value[start_index:]
    return value


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_validation_only":
        raise RuntimeError("验证协议未冻结")
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("验证代码哈希不一致")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}

    dates = arrays["dates"].astype(str)
    base_state = v75.build_base_state(arrays, protocol)
    quality = health_lib.candidate_quality(
        arrays,
        base_state,
        int(protocol["fixed_health"]["shadow_top_n"]),
        float(protocol["health_contract"]["round_trip_cost"]),
    )
    exit_profile = sell_lib.ExitProfile(**protocol["fixed_exit"])
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows = []
    for profile in protocol["validation_profiles"]:
        health = v75.fast_health_exposure(
            quality,
            int(protocol["fixed_health"]["lookback"]),
            float(profile["mean_threshold"]),
            float(profile["positive_fraction"]),
            float(profile["weak_floor"]),
        )
        full_state = v75.state_with_health(base_state, health)
        for requested_start in protocol["empty_start_signal_dates"]:
            indices = np.flatnonzero(dates >= str(requested_start))
            if len(indices) == 0:
                raise RuntimeError(f"找不到启动日: {requested_start}")
            start_index = int(indices[0])
            if start_index + 1 >= len(dates):
                raise RuntimeError(f"启动日后没有买入日: {requested_start}")
            sliced_arrays = {
                key: slice_time(value, start_index, len(dates))
                for key, value in arrays.items()
            }
            sliced_state = (
                full_state[0][start_index:],
                full_state[1][start_index:],
                {key: value[start_index:] for key, value in full_state[2].items()},
                full_state[3][start_index:],
            )
            _, actions = v75.run_case(
                sliced_arrays,
                protocol,
                sliced_state,
                exit_profile,
                record_actions=True,
            )
            case_id = f"{profile['id']}_start_{dates[start_index]}"
            path = action_dir / f"{case_id}.csv"
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            action_rows.append(
                {
                    "case_id": case_id,
                    "profile": profile,
                    "requested_start_signal_date": requested_start,
                    "actual_start_signal_date": str(dates[start_index]),
                    "backtest_start_buy_date": str(dates[start_index + 1]),
                    "backtest_end": "20260720",
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": metrics_lib.digest(path),
                    "rows": len(actions),
                    "buy_rows": int((actions.action == "BUY").sum()),
                    "max_positions": 2,
                    "exit": asdict(exit_profile),
                }
            )
    manifest = {
        "validation_protocol_sha256": metrics_lib.digest(PROTOCOL),
        "actions": action_rows,
        "known_2026_used_for_parameter_selection": False,
        "production_changed": False,
    }
    (OUT / "validation_action_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"validation_paths": len(action_rows), "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
