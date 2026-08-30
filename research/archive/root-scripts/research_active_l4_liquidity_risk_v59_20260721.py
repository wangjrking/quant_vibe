# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_buy_coverage_v39_20260721 as simulator
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_liquidity_risk_v59_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def slot_weights(count, scheme):
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    if scheme == "ratio95":
        return [0.95, 0.05]
    if scheme == "equal":
        return [1.0 / count] * count
    raw = np.arange(count, 0, -1, dtype=float)
    raw /= raw.sum()
    return raw.tolist()


def apply_risk_filter(state, arrays, profile):
    score, order, masked, exposure = state
    masked = {key: value.copy() for key, value in masked.items()}
    atr_pct = np.divide(arrays["atr_qfq"], arrays["close_qfq"], out=np.full_like(arrays["atr_qfq"], np.nan), where=np.isfinite(arrays["close_qfq"]) & (arrays["close_qfq"] > 0))
    valid = np.isfinite(arrays["turnover_rate"])
    valid &= arrays["turnover_rate"] >= profile["turnover_min"]
    valid &= arrays["turnover_rate"] <= profile["turnover_max"]
    valid &= arrays["amount"] >= profile["amount_min"]
    valid &= arrays["total_mv"] >= profile["mv_min"]
    if profile["atr_max"] is not None:
        valid &= np.isfinite(atr_pct) & (atr_pct <= profile["atr_max"])
    masked["signal_clean"] &= valid
    return score, order, masked, exposure


def run_case(arrays, protocol, state, amount_min, mv_min, weight_scheme, exit_profile, record_actions=False):
    buy = simulator.BuyProfile(2, -1.0, amount_min, mv_min, 1.0, False, 0.99, 0.005, 1.0, weight_scheme)
    original = simulator.slot_weights
    simulator.slot_weights = slot_weights
    try:
        return simulator.simulate(arrays, state[2], state[0], state[1], state[3], buy, exit_profile, protocol["observation_end"], record_actions=record_actions)
    finally:
        simulator.slot_weights = original


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

    fixed = p["fixed"]
    base_state = robust.build_state(arrays, fixed["score_weights"], fixed["agreement"], fixed["entry"]["risk_off"], fixed["entry"]["risk_on"], fixed)
    base_exit = sell_lib.ExitProfile(**fixed["exit"])
    rows, filter_defs = [], {}
    g1 = p["stage1_grid"]
    for atr_max in g1["atr_max"]:
        for turnover_min in g1["turnover_min"]:
            for turnover_max in g1["turnover_max"]:
                if turnover_max <= turnover_min:
                    continue
                for amount_min in g1["amount_min"]:
                    for mv_min in g1["mv_min"]:
                        profile = {"atr_max": atr_max, "turnover_min": turnover_min, "turnover_max": turnover_max, "amount_min": amount_min, "mv_min": mv_min}
                        fid = stable_id(profile)
                        filter_defs[fid] = profile
                        state = apply_risk_filter(base_state, arrays, profile)
                        daily = run_case(arrays, p, state, amount_min, mv_min, "ratio95", base_exit)
                        values = robust.evaluate_robust(daily, p)
                        cid = "lr1_" + fid
                        rows.append({"case_id": cid, "filter_id": fid, **profile, **values})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = stage1["robust_positive"] & (stage1["full_max_drawdown"] <= 0.40) & (stage1["full_trades"] >= 80)
    stage1 = rank_results(stage1)
    OUT.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    seeds = stage1[stage1["eligible"]].head(g1["promote_count"])

    rows, stage2_defs = [], {}
    g2 = p["stage2_grid"]
    for _, seed in seeds.iterrows():
        risk_filter = filter_defs[str(seed.filter_id)]
        for weights in g2["score_weights"]:
            for exposure_floor in g2["exposure_floor"]:
                local_fixed = dict(fixed)
                local_fixed["exposure_floor"] = exposure_floor
                state = robust.build_state(arrays, weights, fixed["agreement"], fixed["entry"]["risk_off"], fixed["entry"]["risk_on"], local_fixed)
                state = apply_risk_filter(state, arrays, risk_filter)
                for weight_scheme in g2["weight_scheme"]:
                    for exit_values in g2["exit_profiles"]:
                        exit_profile = sell_lib.ExitProfile("rank_5d", **exit_values)
                        daily = run_case(arrays, p, state, risk_filter["amount_min"], risk_filter["mv_min"], weight_scheme, exit_profile)
                        values = robust.evaluate_robust(daily, p)
                        definition = {"filter_id": seed.filter_id, "weights": weights, "exposure_floor": exposure_floor, "weight_scheme": weight_scheme, "exit": asdict(exit_profile)}
                        cid = "lr2_" + stable_id(definition)
                        stage2_defs[cid] = definition
                        rows.append({"case_id": cid, "filter_id": seed.filter_id, "weights_json": json.dumps(weights, sort_keys=True), "exposure_floor": exposure_floor, "weight_scheme": weight_scheme, **asdict(exit_profile), **values})
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
        definition = stage2_defs[item.case_id]
        risk_filter = filter_defs[str(definition["filter_id"])]
        local_fixed = dict(fixed)
        local_fixed["exposure_floor"] = definition["exposure_floor"]
        state = robust.build_state(arrays, definition["weights"], fixed["agreement"], fixed["entry"]["risk_off"], fixed["entry"]["risk_on"], local_fixed)
        state = apply_risk_filter(state, arrays, risk_filter)
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        _, actions = run_case(arrays, p, state, risk_filter["amount_min"], risk_filter["mv_min"], definition["weight_scheme"], exit_profile, record_actions=True)
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
