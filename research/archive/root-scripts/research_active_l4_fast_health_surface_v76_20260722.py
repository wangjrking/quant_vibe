# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_fast_health_neighborhood_v75_20260722 as v75
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_lagged_health_liquidity_v66_20260721 as health_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_fast_health_surface_v76_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only":
        raise RuntimeError("研究协议未冻结")
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("研究代码哈希不一致")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = robust.truncate_observation(
            {key: saved[key] for key in saved.files}, protocol["observation_end"]
        )

    OUT.mkdir(parents=True, exist_ok=True)
    base_state = v75.build_base_state(arrays, protocol)
    quality = health_lib.candidate_quality(
        arrays,
        base_state,
        int(protocol["fixed_health"]["shadow_top_n"]),
        float(protocol["health_contract"]["round_trip_cost"]),
    )
    exit_profile = sell_lib.ExitProfile(**protocol["fixed_exit"])
    rows = []
    action_payloads = []
    for mean_index, mean_threshold in enumerate(protocol["grid"]["mean_threshold"]):
        for positive_index, positive_fraction in enumerate(protocol["grid"]["positive_fraction"]):
            for floor_index, weak_floor in enumerate(protocol["grid"]["weak_floor"]):
                definition = {
                    "mean_index": mean_index,
                    "positive_index": positive_index,
                    "floor_index": floor_index,
                    "mean_threshold": mean_threshold,
                    "positive_fraction": positive_fraction,
                    "weak_floor": weak_floor,
                }
                case_id = "v76_" + stable_id(definition)
                health = v75.fast_health_exposure(
                    quality,
                    int(protocol["fixed_health"]["lookback"]),
                    float(mean_threshold),
                    float(positive_fraction),
                    float(weak_floor),
                )
                state = v75.state_with_health(base_state, health)
                daily, actions = v75.run_case(
                    arrays, protocol, state, exit_profile, record_actions=True
                )
                values = robust.evaluate_robust(daily, protocol)
                rows.append(
                    {
                        "case_id": case_id,
                        **definition,
                        **asdict(exit_profile),
                        **values,
                    }
                )
                action_payloads.append((case_id, definition, actions))

    results = pd.DataFrame(rows)
    results["eligible"] = v75.local_gate(results, protocol["local_gate"])
    results = results.sort_values(
        ["eligible", "mean_index", "positive_index", "floor_index", "case_id"],
        ascending=[False, True, True, True, True],
    )
    results.to_csv(OUT / "local_surface_results.csv", index=False, encoding="utf-8-sig")

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    by_hash = {}
    eligible_ids = set(results.loc[results.eligible, "case_id"].astype(str))
    for case_id, definition, actions in action_payloads:
        if case_id not in eligible_ids:
            continue
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash not in by_hash:
            path = action_dir / f"{case_id}.csv"
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            by_hash[content_hash] = {
                "representative_case_id": case_id,
                "case_ids": [],
                "grid_points": [],
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": metrics_lib.digest(path),
                "rows": len(actions),
                "buy_rows": int((actions.action == "BUY").sum()),
                "max_positions": 2,
            }
        by_hash[content_hash]["case_ids"].append(case_id)
        by_hash[content_hash]["grid_points"].append(definition)

    manifest = {
        "protocol_sha256": metrics_lib.digest(PROTOCOL),
        "actions": list(by_hash.values()),
    }
    (OUT / "juejin_action_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "research_only_observation_surface_frozen",
        "local_cases": len(results),
        "local_eligible": int(results.eligible.sum()),
        "unique_juejin_paths": len(by_hash),
        "known_2026_used": False,
        "production_changed": False,
    }
    (OUT / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
