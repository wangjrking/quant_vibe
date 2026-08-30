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
import research_active_l4_fast_health_cash_v69_20260721 as fast_health_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_lagged_health_liquidity_v66_20260721 as health_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_1d_veto_v70_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_state(arrays, protocol, score_profile, agreement, entry, rank1_min):
    state = robust.build_state(
        arrays, score_profile["weights"], agreement,
        float(entry["risk_off"]), float(entry["risk_on"]), protocol["fixed"],
    )
    score, order, masked, exposure = state
    masked = {key: value.copy() for key, value in masked.items()}
    valid = np.isfinite(arrays["turnover_rate"])
    valid &= arrays["turnover_rate"] >= 0.0
    valid &= arrays["turnover_rate"] <= float(protocol["fixed"]["turnover_max"])
    valid &= arrays["amount"] >= float(protocol["fixed"]["amount_min"])
    valid &= arrays["total_mv"] >= float(protocol["fixed"]["mv_min"])
    valid &= np.isfinite(arrays["rank_1d"])
    valid &= arrays["rank_1d"] >= float(rank1_min)
    masked["signal_clean"] &= valid
    return score, order, masked, exposure


def state_with_health(arrays, protocol, base_state, mode):
    if mode["id"] == "none":
        return base_state
    quality = health_lib.candidate_quality(
        arrays, base_state, int(mode["shadow_top_n"]),
        float(protocol["health_contract"]["round_trip_cost"]),
    )
    exposure = fast_health_lib.fast_health_exposure(
        quality, int(mode["lookback"]), float(mode["mean_threshold"]),
        float(mode["positive_fraction"]), float(mode["weak_floor"]),
    )
    return (
        base_state[0], base_state[1], base_state[2],
        np.minimum(base_state[3], exposure).astype(np.float32),
    )


def run_case(arrays, protocol, state, exit_profile, record_actions=False):
    fixed = protocol["fixed"]
    buy = simulator.BuyProfile(
        2, -1.0, int(fixed["amount_min"]), int(fixed["mv_min"]),
        1.0, False, 0.99, 0.005, 1.0, "equal",
    )
    return simulator.simulate(
        arrays, state[2], state[0], state[1], state[3], buy, exit_profile,
        protocol["observation_end"], record_actions=record_actions,
    )


def eligible(frame):
    return (
        frame["robust_positive"]
        & (frame["full_max_drawdown"] <= 0.40)
        & (frame["full_trades"] >= 60)
        & (frame["full_invested_ratio_mean"] >= 0.20)
    )


def select_union(frame, robust_count, return_count):
    pool = frame[frame["eligible"]].copy()
    if pool.empty:
        return pool
    robust_top = pool.sort_values(
        ["robust_sharpe_floor", "min_year_sharpe", "recent60_sharpe", "recent120_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, False, True],
    ).head(robust_count)
    return_top = pool.sort_values(
        ["full_cumulative_return", "robust_sharpe_floor", "case_id"],
        ascending=[False, False, True],
    ).head(return_count)
    return pd.concat([robust_top, return_top]).drop_duplicates("case_id")


def write_stop(stage1, reason):
    summary = {
        "status": "research_only_stopped_at_local_gate",
        "stage1_cases": len(stage1),
        "stage1_eligible": int(stage1.eligible.sum()),
        "known_2026_used": False,
        "juejin_run_performed": False,
        "production_changed": False,
        "stop_reason": reason,
    }
    (OUT / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


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
    fixed_exit = sell_lib.ExitProfile(**protocol["stage1_fixed_exit"])
    stage1_rows, stage1_defs, states = [], {}, {}
    for score_profile in protocol["stage1_grid"]["score_profiles"]:
        for agreement in protocol["stage1_grid"]["agreement_profiles"]:
            for entry in protocol["stage1_grid"]["entry_rank_pairs"]:
                for rank1_min in protocol["stage1_grid"]["rank1_min"]:
                    definition = {
                        "score_profile": score_profile,
                        "agreement": agreement,
                        "entry": entry,
                        "rank1_min": rank1_min,
                    }
                    state_key = stable_id(definition)
                    state = build_state(
                        arrays, protocol, score_profile, agreement, entry, rank1_min
                    )
                    states[state_key] = state
                    daily = run_case(arrays, protocol, state, fixed_exit)
                    values = robust.evaluate_robust(daily, protocol)
                    case_id = "iv1_" + stable_id({**definition, "exit": asdict(fixed_exit)})
                    stage1_defs[case_id] = {**definition, "state_key": state_key}
                    stage1_rows.append({
                        "case_id": case_id,
                        "state_key": state_key,
                        "score_profile_id": score_profile["id"],
                        "agreement_id": agreement["id"],
                        "entry_json": json.dumps(entry, sort_keys=True),
                        "rank1_min": rank1_min,
                        **asdict(fixed_exit),
                        **values,
                    })

    stage1 = pd.DataFrame(stage1_rows)
    stage1["eligible"] = eligible(stage1)
    stage1 = stage1.sort_values(
        ["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, True],
    )
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    promoted = select_union(
        stage1, int(protocol["stage1_grid"]["promote_robust_count"]),
        int(protocol["stage1_grid"]["promote_return_count"]),
    )
    if promoted.empty:
        write_stop(stage1, "没有1D否决组合通过第一阶段门槛")
        return

    stage2_rows, stage2_defs = [], {}
    for _, seed in promoted.iterrows():
        definition = stage1_defs[str(seed.case_id)]
        base_state = states[definition["state_key"]]
        for health_mode in protocol["stage2_grid"]["health_modes"]:
            state = state_with_health(arrays, protocol, base_state, health_mode)
            for exit_values in protocol["stage2_grid"]["exit_profiles"]:
                exit_profile = sell_lib.ExitProfile("rank_5d", **exit_values)
                daily = run_case(arrays, protocol, state, exit_profile)
                values = robust.evaluate_robust(daily, protocol)
                case_definition = {
                    "seed_case_id": str(seed.case_id),
                    "health_mode": health_mode,
                    "exit": asdict(exit_profile),
                }
                case_id = "iv2_" + stable_id(case_definition)
                stage2_defs[case_id] = {
                    "seed_definition": definition,
                    **case_definition,
                }
                stage2_rows.append({
                    "case_id": case_id,
                    "seed_case_id": str(seed.case_id),
                    "score_profile_id": definition["score_profile"]["id"],
                    "agreement_id": definition["agreement"]["id"],
                    "rank1_min": definition["rank1_min"],
                    "health_mode_id": health_mode["id"],
                    **asdict(exit_profile),
                    **values,
                })

    stage2 = pd.DataFrame(stage2_rows)
    stage2["eligible"] = eligible(stage2)
    stage2 = stage2.sort_values(
        ["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, True],
    )
    stage2.to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = select_union(
        stage2, int(protocol["stage2_grid"]["freeze_robust_count"]),
        int(protocol["stage2_grid"]["freeze_return_count"]),
    )
    if frozen.empty:
        write_stop(stage1, "没有1D否决与卖出组合通过第二阶段门槛")
        return

    (OUT / "frozen_juejin_candidates.json").write_text(
        json.dumps({
            "protocol_sha256": metrics_lib.digest(PROTOCOL),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "profiles": frozen.to_dict("records"),
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        definition = stage2_defs[str(item.case_id)]
        base_state = states[definition["seed_definition"]["state_key"]]
        state = state_with_health(
            arrays, protocol, base_state, definition["health_mode"]
        )
        exit_profile = sell_lib.ExitProfile(
            str(item.exit_score_source), int(item.min_hold), int(item.max_hold),
            float(item.sell_rank_below), float(item.replacement_advantage),
        )
        _, actions = run_case(
            arrays, protocol, state, exit_profile, record_actions=True
        )
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({
            "case_id": item.case_id,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": metrics_lib.digest(path),
            "rows": len(actions),
            "buy_rows": int((actions.action == "BUY").sum()),
            "max_positions": 2,
        })
    (OUT / "juejin_action_manifest.json").write_text(
        json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "research_only_observation",
        "stage1_cases": len(stage1),
        "stage1_eligible": int(stage1.eligible.sum()),
        "stage1_promoted": len(promoted),
        "stage2_cases": len(stage2),
        "stage2_eligible": int(stage2.eligible.sum()),
        "frozen_candidates": len(frozen),
        "unique_juejin_paths": len(action_rows),
        "known_2026_used": False,
        "production_changed": False,
    }
    (OUT / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
