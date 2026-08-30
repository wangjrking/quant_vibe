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
import research_active_l4_lagged_health_liquidity_v66_20260721 as health_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_fast_health_cash_v69_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def apply_liquidity(state, arrays, fixed):
    score, order, masked, exposure = state
    masked = {key: value.copy() for key, value in masked.items()}
    valid = np.isfinite(arrays["turnover_rate"])
    valid &= arrays["turnover_rate"] >= 0.0
    valid &= arrays["turnover_rate"] <= float(fixed["turnover_max"])
    valid &= arrays["amount"] >= float(fixed["amount_min"])
    valid &= arrays["total_mv"] >= float(fixed["mv_min"])
    masked["signal_clean"] &= valid
    return score, order, masked, exposure


def build_state(arrays, protocol, weights, agreement, entry):
    state = robust.build_state(
        arrays, weights, agreement,
        float(entry["risk_off"]), float(entry["risk_on"]), protocol["fixed"],
    )
    return apply_liquidity(state, arrays, protocol["fixed"])


def fast_health_exposure(quality, lookback, mean_threshold, positive_fraction, weak_floor):
    result = np.full(len(quality), float(weak_floor), dtype=np.float32)
    minimum = max(4, int(np.ceil(lookback * 0.6)))
    for t in range(len(quality)):
        # quality[s]到s+2才成熟，因此t日最多使用s=t-2。
        end = t - 1
        start = max(0, end - int(lookback))
        values = quality[start:end]
        values = values[np.isfinite(values)]
        if len(values) < minimum:
            continue
        healthy = float(np.mean(values)) > float(mean_threshold)
        healthy &= float(np.mean(values > 0.0)) >= float(positive_fraction)
        if healthy:
            result[t] = 1.0
    return result


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


def write_stop_summary(stage1, reason):
    summary = {
        "status": "research_only_stopped_at_local_gate",
        "stage1_cases": len(stage1),
        "stage1_eligible": int(stage1.eligible.sum()),
        "stage2_cases": 0,
        "unique_juejin_paths": 0,
        "known_2026_used": False,
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
    stage1_rows, stage1_defs, state_cache = [], {}, {}
    for weights in protocol["stage1_grid"]["score_weights"]:
        for agreement in protocol["stage1_grid"]["agreement_profiles"]:
            for entry in protocol["stage1_grid"]["entry_rank_pairs"]:
                base_key = stable_id({"weights": weights, "agreement": agreement, "entry": entry})
                base_state = build_state(arrays, protocol, weights, agreement, entry)
                state_cache[base_key] = base_state
                for shadow_top_n in protocol["stage1_grid"]["shadow_top_n"]:
                    quality = health_lib.candidate_quality(
                        arrays, base_state, int(shadow_top_n),
                        float(protocol["health_contract"]["round_trip_cost"]),
                    )
                    for lookback in protocol["stage1_grid"]["lookback"]:
                        for mean_threshold in protocol["stage1_grid"]["mean_threshold"]:
                            for positive_fraction in protocol["stage1_grid"]["positive_fraction"]:
                                for weak_floor in protocol["stage1_grid"]["weak_floor"]:
                                    health = fast_health_exposure(
                                        quality, lookback, mean_threshold,
                                        positive_fraction, weak_floor,
                                    )
                                    state = (
                                        base_state[0], base_state[1], base_state[2],
                                        np.minimum(base_state[3], health).astype(np.float32),
                                    )
                                    daily = run_case(arrays, protocol, state, fixed_exit)
                                    values = robust.evaluate_robust(daily, protocol)
                                    definition = {
                                        "base_key": base_key,
                                        "weights": weights,
                                        "agreement": agreement,
                                        "entry": entry,
                                        "shadow_top_n": shadow_top_n,
                                        "lookback": lookback,
                                        "mean_threshold": mean_threshold,
                                        "positive_fraction": positive_fraction,
                                        "weak_floor": weak_floor,
                                        "exit": asdict(fixed_exit),
                                    }
                                    case_id = "fh1_" + stable_id(definition)
                                    stage1_defs[case_id] = definition
                                    stage1_rows.append({
                                        "case_id": case_id,
                                        "base_key": base_key,
                                        "weights_json": json.dumps(weights, sort_keys=True),
                                        "agreement_id": agreement["id"],
                                        "entry_json": json.dumps(entry, sort_keys=True),
                                        "shadow_top_n": shadow_top_n,
                                        "lookback": lookback,
                                        "mean_threshold": mean_threshold,
                                        "positive_fraction": positive_fraction,
                                        "weak_floor": weak_floor,
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
        stage1,
        int(protocol["stage1_grid"]["promote_robust_count"]),
        int(protocol["stage1_grid"]["promote_return_count"]),
    )
    if promoted.empty:
        write_stop_summary(stage1, "没有快速健康度组合通过预注册第一阶段门槛")
        return

    stage2_rows, stage2_defs = [], {}
    for _, seed in promoted.iterrows():
        definition = stage1_defs[str(seed.case_id)]
        base_state = state_cache[definition["base_key"]]
        quality = health_lib.candidate_quality(
            arrays, base_state, int(definition["shadow_top_n"]),
            float(protocol["health_contract"]["round_trip_cost"]),
        )
        health = fast_health_exposure(
            quality, definition["lookback"], definition["mean_threshold"],
            definition["positive_fraction"], definition["weak_floor"],
        )
        state = (
            base_state[0], base_state[1], base_state[2],
            np.minimum(base_state[3], health).astype(np.float32),
        )
        for exit_values in protocol["stage2_grid"]["exit_profiles"]:
            exit_profile = sell_lib.ExitProfile("rank_5d", **exit_values)
            daily = run_case(arrays, protocol, state, exit_profile)
            values = robust.evaluate_robust(daily, protocol)
            case_definition = {
                "seed_case_id": str(seed.case_id),
                "exit": asdict(exit_profile),
            }
            case_id = "fh2_" + stable_id(case_definition)
            stage2_defs[case_id] = {"seed_definition": definition, **case_definition}
            stage2_rows.append({
                "case_id": case_id,
                "seed_case_id": str(seed.case_id),
                "agreement_id": definition["agreement"]["id"],
                "shadow_top_n": definition["shadow_top_n"],
                "lookback": definition["lookback"],
                "mean_threshold": definition["mean_threshold"],
                "positive_fraction": definition["positive_fraction"],
                "weak_floor": definition["weak_floor"],
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
        stage2,
        int(protocol["stage2_grid"]["freeze_robust_count"]),
        int(protocol["stage2_grid"]["freeze_return_count"]),
    )
    if frozen.empty:
        write_stop_summary(stage1, "第二阶段卖出参数没有候选通过门槛")
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
        definition = stage2_defs[str(item.case_id)]["seed_definition"]
        base_state = state_cache[definition["base_key"]]
        quality = health_lib.candidate_quality(
            arrays, base_state, int(definition["shadow_top_n"]),
            float(protocol["health_contract"]["round_trip_cost"]),
        )
        health = fast_health_exposure(
            quality, definition["lookback"], definition["mean_threshold"],
            definition["positive_fraction"], definition["weak_floor"],
        )
        state = (
            base_state[0], base_state[1], base_state[2],
            np.minimum(base_state[3], health).astype(np.float32),
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
