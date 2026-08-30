# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_buy_coverage_v39_20260721 as simulator
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_lagged_health_liquidity_v66_20260721 as health_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_fast_health_neighborhood_v75_20260722"
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


def build_base_state(arrays, protocol):
    state = robust.build_state(
        arrays,
        protocol["fixed_score_weights"],
        protocol["fixed_agreement"],
        float(protocol["fixed_entry"]["risk_off"]),
        float(protocol["fixed_entry"]["risk_on"]),
        protocol["fixed"],
    )
    return apply_liquidity(state, arrays, protocol["fixed"])


def fast_health_exposure(quality, lookback, mean_threshold, positive_fraction, weak_floor):
    result = np.full(len(quality), float(weak_floor), dtype=np.float32)
    minimum = max(4, int(np.ceil(lookback * 0.6)))
    for t in range(len(quality)):
        # quality[s] is only mature at s+2, so T may use at most s=T-2.
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


def state_with_health(base_state, health):
    return (
        base_state[0],
        base_state[1],
        base_state[2],
        np.minimum(base_state[3], health).astype(np.float32),
    )


def run_case(arrays, protocol, state, exit_profile, record_actions=False):
    fixed = protocol["fixed"]
    buy = simulator.BuyProfile(
        2,
        -1.0,
        int(fixed["amount_min"]),
        int(fixed["mv_min"]),
        1.0,
        False,
        0.99,
        0.005,
        1.0,
        "equal",
    )
    return simulator.simulate(
        arrays,
        state[2],
        state[0],
        state[1],
        state[3],
        buy,
        exit_profile,
        protocol["observation_end"],
        record_actions=record_actions,
    )


def local_gate(frame, gate):
    return (
        frame["robust_positive"]
        & (frame["full_linear_annual_proxy"] >= float(gate["linear_annual_proxy_min"]))
        & (frame["full_sharpe"] >= float(gate["sharpe_min"]))
        & (frame["full_max_drawdown"] <= float(gate["max_drawdown_max"]))
        & (frame["full_trades"] >= int(gate["trades_min"]))
        & (frame["full_invested_ratio_mean"] >= float(gate["invested_ratio_min"]))
        & (frame["min_year_return"] > 0.0)
        & (frame["recent60_cumulative_return"] > 0.0)
        & (frame["recent120_cumulative_return"] > 0.0)
    )


def rank_candidates(frame, count):
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["target_distance"] = (
        np.maximum(0.0, 2.0 - ranked["full_linear_annual_proxy"])
        + 2.0 * np.maximum(0.0, 2.0 - ranked["full_sharpe"])
    )
    return ranked.sort_values(
        [
            "target_distance",
            "full_sharpe",
            "full_linear_annual_proxy",
            "robust_sharpe_floor",
            "case_id",
        ],
        ascending=[True, False, False, False, True],
    ).head(int(count))


def write_summary(payload):
    (OUT / "research_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


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
    base_state = build_base_state(arrays, protocol)
    fixed_exit = sell_lib.ExitProfile(**protocol["stage1_fixed_exit"])
    health_cache = {}
    stage1_rows = []
    grid = protocol["stage1_grid"]
    for shadow_top_n in grid["shadow_top_n"]:
        quality = health_lib.candidate_quality(
            arrays,
            base_state,
            int(shadow_top_n),
            float(protocol["health_contract"]["round_trip_cost"]),
        )
        for lookback in grid["lookback"]:
            for mean_threshold in grid["mean_threshold"]:
                for positive_fraction in grid["positive_fraction"]:
                    for weak_floor in grid["weak_floor"]:
                        definition = {
                            "shadow_top_n": shadow_top_n,
                            "lookback": lookback,
                            "mean_threshold": mean_threshold,
                            "positive_fraction": positive_fraction,
                            "weak_floor": weak_floor,
                        }
                        case_id = "v75h_" + stable_id(definition)
                        health = fast_health_exposure(
                            quality,
                            lookback,
                            mean_threshold,
                            positive_fraction,
                            weak_floor,
                        )
                        health_cache[case_id] = health
                        daily = run_case(
                            arrays,
                            protocol,
                            state_with_health(base_state, health),
                            fixed_exit,
                        )
                        stage1_rows.append(
                            {
                                "case_id": case_id,
                                **definition,
                                **asdict(fixed_exit),
                                **robust.evaluate_robust(daily, protocol),
                            }
                        )

    stage1 = pd.DataFrame(stage1_rows)
    stage1["eligible"] = local_gate(stage1, protocol["stage1_gate"])
    stage1 = stage1.sort_values(
        ["eligible", "full_sharpe", "full_linear_annual_proxy", "case_id"],
        ascending=[False, False, False, True],
    )
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    promoted = rank_candidates(
        stage1[stage1["eligible"]], protocol["stage1_grid"]["promote_count"]
    )
    if promoted.empty:
        write_summary(
            {
                "status": "research_only_stopped_at_stage1_gate",
                "stage1_cases": len(stage1),
                "stage1_eligible": 0,
                "known_2026_used": False,
                "production_changed": False,
            }
        )
        return

    stage2_rows = []
    stage2_definitions = {}
    for _, seed in promoted.iterrows():
        health = health_cache[str(seed.case_id)]
        state = state_with_health(base_state, health)
        exit_grid = protocol["stage2_grid"]
        exit_values = product(
            exit_grid["min_hold"],
            exit_grid["max_hold"],
            exit_grid["sell_rank_below"],
            exit_grid["replacement_advantage"],
        )
        for min_hold, max_hold, sell_rank_below, replacement_advantage in exit_values:
            if int(max_hold) < int(min_hold):
                continue
            values = {
                "min_hold": int(min_hold),
                "max_hold": int(max_hold),
                "sell_rank_below": float(sell_rank_below),
                "replacement_advantage": float(replacement_advantage),
            }
            exit_profile = sell_lib.ExitProfile("rank_5d", **values)
            definition = {
                "health_case_id": str(seed.case_id),
                "shadow_top_n": int(seed.shadow_top_n),
                "lookback": int(seed.lookback),
                "mean_threshold": float(seed.mean_threshold),
                "positive_fraction": float(seed.positive_fraction),
                "weak_floor": float(seed.weak_floor),
                "exit": asdict(exit_profile),
            }
            case_id = "v75e_" + stable_id(definition)
            daily = run_case(arrays, protocol, state, exit_profile)
            stage2_definitions[case_id] = definition
            stage2_rows.append(
                {
                    "case_id": case_id,
                    "health_case_id": str(seed.case_id),
                    "shadow_top_n": int(seed.shadow_top_n),
                    "lookback": int(seed.lookback),
                    "mean_threshold": float(seed.mean_threshold),
                    "positive_fraction": float(seed.positive_fraction),
                    "weak_floor": float(seed.weak_floor),
                    **asdict(exit_profile),
                    **robust.evaluate_robust(daily, protocol),
                }
            )

    stage2 = pd.DataFrame(stage2_rows)
    stage2["eligible"] = local_gate(stage2, protocol["stage2_gate"])
    stage2 = stage2.sort_values(
        ["eligible", "full_sharpe", "full_linear_annual_proxy", "case_id"],
        ascending=[False, False, False, True],
    )
    stage2.to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = rank_candidates(
        stage2[stage2["eligible"]], protocol["stage2_grid"]["freeze_count"]
    )
    if frozen.empty:
        write_summary(
            {
                "status": "research_only_stopped_at_stage2_gate",
                "stage1_cases": len(stage1),
                "stage1_eligible": int(stage1.eligible.sum()),
                "stage2_cases": len(stage2),
                "stage2_eligible": 0,
                "known_2026_used": False,
                "production_changed": False,
            }
        )
        return

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows = []
    seen = set()
    for _, item in frozen.iterrows():
        definition = stage2_definitions[str(item.case_id)]
        health = health_cache[definition["health_case_id"]]
        state = state_with_health(base_state, health)
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        _, actions = run_case(
            arrays, protocol, state, exit_profile, record_actions=True
        )
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append(
            {
                "case_id": str(item.case_id),
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": metrics_lib.digest(path),
                "rows": len(actions),
                "buy_rows": int((actions.action == "BUY").sum()),
                "max_positions": 2,
                "local_metrics": {
                    "linear_annual_proxy": float(item.full_linear_annual_proxy),
                    "sharpe": float(item.full_sharpe),
                    "max_drawdown": float(item.full_max_drawdown),
                },
            }
        )
    (OUT / "frozen_candidates.json").write_text(
        json.dumps(
            {
                "protocol_sha256": metrics_lib.digest(PROTOCOL),
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "profiles": frozen.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "juejin_action_manifest.json").write_text(
        json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(
        {
            "status": "research_only_observation_candidates_frozen",
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
    )


if __name__ == "__main__":
    main()
