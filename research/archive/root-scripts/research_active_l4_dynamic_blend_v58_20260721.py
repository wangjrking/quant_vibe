# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement_lib
import research_active_l4_buy_coverage_v39_20260721 as simulator
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_continuous_market_v53_20260721 as exposure_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_market_state_v26_20260721 as market_lib
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_weak_open_refine_v49_20260721 as weak_open
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_dynamic_blend_v58_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def dynamic_state(arrays, off_weights, on_weights, threshold, agreement, entry, exposure_floor, fixed):
    off_score = core.blend_scores(arrays, off_weights)
    on_score = core.blend_scores(arrays, on_weights)
    market_mean = exposure_lib.rolling_mean(market_lib.market_open_return(arrays), fixed["market_lookback"])
    market_on = np.isfinite(market_mean) & (market_mean > threshold)
    score = np.where(market_on[:, None], on_score, off_score).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement_lib.masked_arrays(arrays, agreement)
    masked = weak_open.gated_mask(masked, arrays, fixed["min_open_gap"])
    dynamic_entry = np.where(market_on, entry["risk_on"], entry["risk_off"]).astype(np.float32)
    masked["signal_clean"] &= score >= dynamic_entry[:, None]
    exposure = exposure_lib.continuous_exposure(market_mean, exposure_floor, fixed["linear_low_mean_return"], fixed["linear_high_mean_return"])
    return score, order, masked, exposure


def run_case(arrays, state, fixed, exit_profile, weight_scheme, end_date, record_actions=False):
    buy_profile = simulator.BuyProfile(2, -1.0, fixed["amount_min"], fixed["mv_min"], 1.0, False, 0.99, 0.005, 1.0, weight_scheme)
    return simulator.simulate(arrays, state[2], state[0], state[1], state[3], buy_profile, exit_profile, end_date, record_actions=record_actions)


def rank_results(frame):
    return frame.sort_values(
        ["eligible", "robust_sharpe_floor", "min_year_sharpe", "recent60_sharpe", "recent120_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, False, False, True],
    )


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / p["input_cache"]["path"]
    if p["status"] != "frozen_research_only" or metrics_lib.digest(cache) != p["input_cache"]["sha256"] or metrics_lib.digest(Path(__file__)) != p["code_sha256"]:
        raise RuntimeError("frozen protocol, code, or input mismatch")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = robust.truncate_observation({key: saved[key] for key in saved.files}, p["observation_end"])

    base_exit = sell_lib.ExitProfile(**p["stage1_fixed_exit"])
    states, definitions, rows = {}, {}, []
    g1 = p["stage1_grid"]
    for off_weights in g1["risk_off_weights"]:
        for on_weights in g1["risk_on_weights"]:
            for threshold in g1["market_threshold"]:
                for agreement in g1["agreement_profiles"]:
                    for entry in g1["entry_rank_pairs"]:
                        definition = {"off_weights": off_weights, "on_weights": on_weights, "threshold": threshold, "agreement": agreement, "entry": entry}
                        state_key = stable_id(definition)
                        state = dynamic_state(arrays, off_weights, on_weights, threshold, agreement, entry, p["fixed"]["stage1_exposure_floor"], p["fixed"])
                        states[state_key] = state
                        definitions[state_key] = definition
                        daily = run_case(arrays, state, p["fixed"], base_exit, "equal", p["observation_end"])
                        values = robust.evaluate_robust(daily, p)
                        cid = "db1_" + stable_id({"state": state_key, "exit": asdict(base_exit)})
                        rows.append({"case_id": cid, "state_key": state_key, "definition_json": json.dumps(definition, sort_keys=True), **asdict(base_exit), **values})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = stage1["robust_positive"] & (stage1["full_max_drawdown"] <= 0.40) & (stage1["full_trades"] >= 80)
    stage1 = rank_results(stage1)
    OUT.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    seeds = stage1[stage1["eligible"]].head(g1["promote_count"])

    rows, stage2_defs = [], {}
    g2 = p["stage2_grid"]
    for _, seed in seeds.iterrows():
        definition = definitions[str(seed.state_key)]
        for exposure_floor in g2["exposure_floor"]:
            state = dynamic_state(arrays, definition["off_weights"], definition["on_weights"], definition["threshold"], definition["agreement"], definition["entry"], exposure_floor, p["fixed"])
            for weight_scheme in g2["weight_scheme"]:
                for exit_values in g2["exit_profiles"]:
                    exit_profile = sell_lib.ExitProfile("rank_5d", **exit_values)
                    daily = run_case(arrays, state, p["fixed"], exit_profile, weight_scheme, p["observation_end"])
                    values = robust.evaluate_robust(daily, p)
                    payload = {"seed": seed.case_id, "exposure_floor": exposure_floor, "weight_scheme": weight_scheme, "exit": asdict(exit_profile)}
                    cid = "db2_" + stable_id(payload)
                    stage2_defs[cid] = {"state_definition": definition, **payload}
                    rows.append({"case_id": cid, "seed_case_id": seed.case_id, "exposure_floor": exposure_floor, "weight_scheme": weight_scheme, **asdict(exit_profile), **values})
    stage2 = pd.DataFrame(rows)
    if stage2.empty:
        stage2 = pd.DataFrame(columns=["case_id", "eligible"])
        frozen = stage2.copy()
    else:
        stage2["eligible"] = stage2["robust_positive"] & (stage2["full_max_drawdown"] <= 0.40) & (stage2["full_trades"] >= 80)
        stage2 = rank_results(stage2)
        frozen = stage2[stage2["eligible"]].head(g2["promote_count"])
    stage2.to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": metrics_lib.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        item_def = stage2_defs[item.case_id]
        definition = item_def["state_definition"]
        state = dynamic_state(arrays, definition["off_weights"], definition["on_weights"], definition["threshold"], definition["agreement"], definition["entry"], item_def["exposure_floor"], p["fixed"])
        exit_profile = sell_lib.ExitProfile(**item_def["exit"])
        _, actions = run_case(arrays, state, p["fixed"], exit_profile, item_def["weight_scheme"], p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": metrics_lib.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_eligible": int(stage1.eligible.sum()), "stage1_promoted": len(seeds), "stage2_cases": len(stage2), "stage2_eligible": int(stage2.eligible.sum()) if "eligible" in stage2 else 0, "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
