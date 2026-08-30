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
import research_active_l4_lagged_health_liquidity_v66_20260721 as health_lib
import research_active_l4_market_state_v26_20260721 as market_lib
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_weak_open_refine_v49_20260721 as weak_open
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_sustainable_portfolio_v68_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def percentile_rank(values):
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for t in range(values.shape[0]):
        valid = np.isfinite(values[t])
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        order = np.argsort(values[t, indices], kind="mergesort")
        ranks = np.empty(len(indices), dtype=np.float32)
        if len(indices) == 1:
            ranks[order] = 1.0
        else:
            ranks[order] = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
        result[t, indices] = ranks
    return result


def build_state(arrays, protocol, score_profile, agreement, entry_min, liquidity):
    base = core.blend_scores(arrays, score_profile["weights"])
    atr_pct = np.divide(
        arrays["atr_qfq"], arrays["close_qfq"],
        out=np.full_like(arrays["atr_qfq"], np.nan, dtype=np.float32),
        where=np.isfinite(arrays["close_qfq"]) & (arrays["close_qfq"] > 0),
    )
    atr_rank = percentile_rank(atr_pct)
    amount_rank = percentile_rank(arrays["amount"])
    adjusted = (
        base
        - float(score_profile["atr_penalty"]) * np.nan_to_num(atr_rank, nan=1.0)
        + float(score_profile["amount_bonus"]) * np.nan_to_num(amount_rank, nan=0.0)
    )
    buy_score = percentile_rank(adjusted)
    order = np.argsort(-np.nan_to_num(buy_score, nan=-np.inf), axis=1).astype(np.int32)

    masked = agreement_lib.masked_arrays(arrays, agreement)
    masked = weak_open.gated_mask(masked, arrays, protocol["fixed"]["min_open_gap"])
    liquid = np.isfinite(arrays["turnover_rate"])
    liquid &= arrays["turnover_rate"] >= 0.0
    liquid &= arrays["turnover_rate"] <= float(liquidity["turnover_max"])
    liquid &= arrays["amount"] >= float(liquidity["amount_min"])
    liquid &= arrays["total_mv"] >= float(liquidity["mv_min"])
    masked["signal_clean"] &= liquid & (buy_score >= float(entry_min))

    fixed = protocol["fixed"]
    market_mean = exposure_lib.rolling_mean(
        market_lib.market_open_return(arrays), int(fixed["market_lookback"])
    )
    market_exposure = exposure_lib.continuous_exposure(
        market_mean,
        float(fixed["market_exposure_floor"]),
        float(fixed["linear_low_mean_return"]),
        float(fixed["linear_high_mean_return"]),
    )
    return buy_score, order, masked, market_exposure


def run_case(arrays, protocol, state, top_n, exit_profile, record_actions=False):
    liquidity = protocol["runtime_liquidity"]
    profile = simulator.BuyProfile(
        int(top_n), -1.0,
        int(liquidity["amount_min"]), int(liquidity["mv_min"]),
        1.0, False, 0.99, 0.005, 1.0, "equal",
    )
    return simulator.simulate(
        arrays, state[2], state[0], state[1], state[3], profile, exit_profile,
        protocol["observation_end"], record_actions=record_actions,
    )


def eligible(frame):
    return (
        frame["robust_positive"]
        & (frame["full_max_drawdown"] <= 0.40)
        & (frame["full_trades"] >= 80)
        & (frame["full_invested_ratio_mean"] >= 0.30)
    )


def select_union(frame, robust_count, return_count):
    pool = frame[frame["eligible"]].copy()
    robust_top = pool.sort_values(
        ["robust_sharpe_floor", "min_year_sharpe", "recent60_sharpe", "recent120_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, False, True],
    ).head(robust_count)
    return_top = pool.sort_values(
        ["full_cumulative_return", "robust_sharpe_floor", "case_id"],
        ascending=[False, False, True],
    ).head(return_count)
    return pd.concat([robust_top, return_top]).drop_duplicates("case_id")


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
    base_exit = sell_lib.ExitProfile(**protocol["stage1_fixed_exit"])
    stage1_rows, stage1_defs, stage1_states = [], {}, {}
    for score_profile in protocol["stage1_grid"]["score_profiles"]:
        for agreement in protocol["stage1_grid"]["agreement_profiles"]:
            for entry_min in protocol["stage1_grid"]["entry_rank_min"]:
                for liquidity in protocol["stage1_grid"]["liquidity_profiles"]:
                    state_definition = {
                        "score_profile": score_profile,
                        "agreement": agreement,
                        "entry_rank_min": entry_min,
                        "liquidity": liquidity,
                    }
                    state_key = stable_id(state_definition)
                    state = build_state(
                        arrays, protocol, score_profile, agreement, entry_min, liquidity
                    )
                    stage1_states[state_key] = state
                    for top_n in protocol["stage1_grid"]["top_n"]:
                        local_protocol = dict(protocol)
                        local_protocol["runtime_liquidity"] = liquidity
                        daily = run_case(
                            arrays, local_protocol, state, top_n, base_exit
                        )
                        values = robust.evaluate_robust(daily, protocol)
                        definition = {
                            **state_definition,
                            "state_key": state_key,
                            "top_n": top_n,
                            "exit": asdict(base_exit),
                        }
                        case_id = "sp1_" + stable_id(definition)
                        stage1_defs[case_id] = definition
                        stage1_rows.append({
                            "case_id": case_id,
                            "state_key": state_key,
                            "score_profile_id": score_profile["id"],
                            "agreement_id": agreement["id"],
                            "entry_rank_min": entry_min,
                            "liquidity_id": liquidity["id"],
                            "top_n": top_n,
                            **asdict(base_exit),
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

    stage2_rows, stage2_defs = [], {}
    for _, seed in promoted.iterrows():
        seed_def = stage1_defs[str(seed.case_id)]
        base_state = stage1_states[seed_def["state_key"]]
        quality = health_lib.candidate_quality(
            arrays, base_state, int(seed_def["top_n"]),
            float(protocol["health_contract"]["round_trip_cost"]),
        )
        for lookback in protocol["stage2_grid"]["health_lookback"]:
            for threshold in protocol["stage2_grid"]["health_threshold"]:
                for weak_floor in protocol["stage2_grid"]["health_weak_floor"]:
                    health_exposure = health_lib.lagged_health_exposure(
                        quality, int(lookback), float(threshold), float(weak_floor)
                    )
                    state = (
                        base_state[0], base_state[1], base_state[2],
                        np.minimum(base_state[3], health_exposure).astype(np.float32),
                    )
                    for exit_values in protocol["stage2_grid"]["exit_profiles"]:
                        exit_profile = sell_lib.ExitProfile("rank_5d", **exit_values)
                        local_protocol = dict(protocol)
                        local_protocol["runtime_liquidity"] = seed_def["liquidity"]
                        daily = run_case(
                            arrays, local_protocol, state, seed_def["top_n"], exit_profile
                        )
                        values = robust.evaluate_robust(daily, protocol)
                        definition = {
                            "seed_case_id": str(seed.case_id),
                            "lookback": lookback,
                            "threshold": threshold,
                            "weak_floor": weak_floor,
                            "exit": asdict(exit_profile),
                        }
                        case_id = "sp2_" + stable_id(definition)
                        stage2_defs[case_id] = {
                            "seed_definition": seed_def,
                            **definition,
                        }
                        stage2_rows.append({
                            "case_id": case_id,
                            "seed_case_id": str(seed.case_id),
                            "score_profile_id": seed_def["score_profile"]["id"],
                            "agreement_id": seed_def["agreement"]["id"],
                            "entry_rank_min": seed_def["entry_rank_min"],
                            "liquidity_id": seed_def["liquidity"]["id"],
                            "top_n": seed_def["top_n"],
                            "health_lookback": lookback,
                            "health_threshold": threshold,
                            "health_weak_floor": weak_floor,
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
        seed_def = definition["seed_definition"]
        base_state = stage1_states[seed_def["state_key"]]
        quality = health_lib.candidate_quality(
            arrays, base_state, int(seed_def["top_n"]),
            float(protocol["health_contract"]["round_trip_cost"]),
        )
        health_exposure = health_lib.lagged_health_exposure(
            quality, int(definition["lookback"]), float(definition["threshold"]),
            float(definition["weak_floor"]),
        )
        state = (
            base_state[0], base_state[1], base_state[2],
            np.minimum(base_state[3], health_exposure).astype(np.float32),
        )
        local_protocol = dict(protocol)
        local_protocol["runtime_liquidity"] = seed_def["liquidity"]
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        _, actions = run_case(
            arrays, local_protocol, state, seed_def["top_n"], exit_profile,
            record_actions=True,
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
            "max_positions": int(seed_def["top_n"]),
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
