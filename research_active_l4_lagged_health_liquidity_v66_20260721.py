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
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_lagged_health_liquidity_v66_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def apply_filter(state, arrays, fixed):
    score, order, masked, exposure = state
    masked = {key: value.copy() for key, value in masked.items()}
    valid = np.isfinite(arrays["turnover_rate"])
    valid &= arrays["turnover_rate"] >= 0.0
    valid &= arrays["turnover_rate"] <= fixed["turnover_max"]
    valid &= arrays["amount"] >= fixed["amount_min"]
    valid &= arrays["total_mv"] >= fixed["mv_min"]
    masked["signal_clean"] &= valid
    return score, order, masked, exposure


def build_state(arrays, fixed, weights, agreement, entry):
    state = robust.build_state(
        arrays,
        weights,
        agreement,
        entry["risk_off"],
        entry["risk_on"],
        fixed,
    )
    return apply_filter(state, arrays, fixed)


def candidate_quality(arrays, state, shadow_top_n, round_trip_cost):
    score, order, masked, _ = state
    opens = arrays["buy_open"]
    result = np.full(len(opens), np.nan, dtype=np.float32)
    for s in range(len(opens) - 1):
        valid = (
            masked["signal_clean"][s]
            & masked["buy_clean"][s]
            & np.isfinite(score[s])
            & np.isfinite(opens[s])
            & (opens[s] > 0)
            & np.isfinite(opens[s + 1])
            & (opens[s + 1] > 0)
        )
        chosen = [int(idx) for idx in order[s] if valid[int(idx)]][:shadow_top_n]
        if chosen:
            result[s] = float(
                np.mean(opens[s + 1, chosen] / opens[s, chosen] - 1.0)
                - round_trip_cost
            )
    return result


def lagged_health_exposure(quality, lookback, threshold, weak_floor):
    result = np.full(len(quality), weak_floor, dtype=np.float32)
    minimum = max(5, lookback // 4)
    for t in range(len(quality)):
        # quality[s] matures at signal index s+2. At t, s=t-2 is the newest usable item.
        end = t - 1
        start = max(0, end - lookback)
        values = quality[start:end]
        values = values[np.isfinite(values)]
        if len(values) >= minimum and float(np.mean(values)) > threshold:
            result[t] = 1.0
    return result


def run_case(arrays, protocol, state, exit_profile, record_actions=False):
    fixed = protocol["fixed"]
    buy = simulator.BuyProfile(
        2,
        -1.0,
        fixed["amount_min"],
        fixed["mv_min"],
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


def eligible(frame):
    return (
        frame["robust_positive"]
        & (frame["full_max_drawdown"] <= 0.40)
        & (frame["full_trades"] >= 80)
    )


def select_union(frame, robust_count, return_count):
    pool = frame[frame["eligible"]].copy()
    robust_top = pool.sort_values(
        ["robust_sharpe_floor", "min_year_sharpe", "recent60_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, True],
    ).head(robust_count)
    return_top = pool.sort_values(
        ["full_cumulative_return", "robust_sharpe_floor", "case_id"],
        ascending=[False, False, True],
    ).head(return_count)
    return pd.concat([robust_top, return_top]).drop_duplicates("case_id")


def state_with_health(arrays, protocol, definition):
    fixed = protocol["fixed"]
    state = build_state(
        arrays,
        fixed,
        definition["weights"],
        definition["agreement"],
        definition["entry"],
    )
    quality = candidate_quality(
        arrays,
        state,
        int(definition["shadow_top_n"]),
        protocol["health_contract"]["round_trip_cost"],
    )
    health_exposure = lagged_health_exposure(
        quality,
        int(definition["lookback"]),
        float(definition["threshold"]),
        float(definition["weak_floor"]),
    )
    exposure = np.minimum(state[3], health_exposure).astype(np.float32)
    return state[0], state[1], state[2], exposure


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / p["input_cache"]["path"]
    if p["status"] != "frozen_research_only":
        raise RuntimeError("protocol is not frozen")
    if metrics_lib.digest(cache) != p["input_cache"]["sha256"]:
        raise RuntimeError("input cache hash mismatch")
    if metrics_lib.digest(Path(__file__)) != p["code_sha256"]:
        raise RuntimeError("code hash mismatch")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = robust.truncate_observation(
            {key: saved[key] for key in saved.files}, p["observation_end"]
        )

    base_exit = sell_lib.ExitProfile(**p["stage1_fixed_exit"])
    stage1_rows, stage1_defs = [], {}
    for weights in p["stage1_grid"]["score_weights"]:
        for agreement in p["stage1_grid"]["agreement_profiles"]:
            for shadow_top_n in p["stage1_grid"]["shadow_top_n"]:
                base_definition = {
                    "weights": weights,
                    "agreement": agreement,
                    "entry": p["stage1_grid"]["entry"],
                    "shadow_top_n": shadow_top_n,
                }
                base_state = build_state(
                    arrays,
                    p["fixed"],
                    weights,
                    agreement,
                    p["stage1_grid"]["entry"],
                )
                quality = candidate_quality(
                    arrays,
                    base_state,
                    shadow_top_n,
                    p["health_contract"]["round_trip_cost"],
                )
                for lookback in p["stage1_grid"]["lookback"]:
                    for threshold in p["stage1_grid"]["threshold"]:
                        for weak_floor in p["stage1_grid"]["weak_floor"]:
                            health_exposure = lagged_health_exposure(
                                quality, lookback, threshold, weak_floor
                            )
                            state = (
                                base_state[0],
                                base_state[1],
                                base_state[2],
                                np.minimum(base_state[3], health_exposure).astype(np.float32),
                            )
                            daily = run_case(arrays, p, state, base_exit)
                            values = robust.evaluate_robust(daily, p)
                            definition = {
                                **base_definition,
                                "lookback": lookback,
                                "threshold": threshold,
                                "weak_floor": weak_floor,
                                "exit": asdict(base_exit),
                            }
                            case_id = "lh1_" + stable_id(definition)
                            stage1_defs[case_id] = definition
                            stage1_rows.append({
                                "case_id": case_id,
                                "weights_json": json.dumps(weights, sort_keys=True),
                                "agreement_json": json.dumps(agreement, sort_keys=True),
                                "shadow_top_n": shadow_top_n,
                                "lookback": lookback,
                                "threshold": threshold,
                                "weak_floor": weak_floor,
                                **asdict(base_exit),
                                **values,
                            })

    stage1 = pd.DataFrame(stage1_rows)
    stage1["eligible"] = eligible(stage1)
    stage1 = stage1.sort_values(
        ["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, True],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    promoted = select_union(
        stage1,
        p["stage1_grid"]["promote_robust_count"],
        p["stage1_grid"]["promote_return_count"],
    )

    stage2_rows, stage2_defs = [], {}
    for _, seed in promoted.iterrows():
        seed_def = stage1_defs[str(seed.case_id)]
        for entry in p["stage2_grid"]["entry_rank_pairs"]:
            definition_base = dict(seed_def)
            definition_base["entry"] = entry
            state = state_with_health(arrays, p, definition_base)
            for min_hold in p["stage2_grid"]["min_hold"]:
                for max_hold in p["stage2_grid"]["max_hold"]:
                    if max_hold <= min_hold:
                        continue
                    for sell_rank_below in p["stage2_grid"]["sell_rank_below"]:
                        exit_profile = sell_lib.ExitProfile(
                            "rank_5d",
                            min_hold,
                            max_hold,
                            sell_rank_below,
                            p["stage2_grid"]["replacement_advantage"],
                        )
                        daily = run_case(arrays, p, state, exit_profile)
                        values = robust.evaluate_robust(daily, p)
                        definition = {
                            **definition_base,
                            "exit": asdict(exit_profile),
                        }
                        case_id = "lh2_" + stable_id(definition)
                        stage2_defs[case_id] = definition
                        stage2_rows.append({
                            "case_id": case_id,
                            "seed_case_id": str(seed.case_id),
                            "weights_json": json.dumps(definition["weights"], sort_keys=True),
                            "agreement_json": json.dumps(definition["agreement"], sort_keys=True),
                            "entry_json": json.dumps(entry, sort_keys=True),
                            "shadow_top_n": definition["shadow_top_n"],
                            "lookback": definition["lookback"],
                            "threshold": definition["threshold"],
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
        p["stage2_grid"]["freeze_robust_count"],
        p["stage2_grid"]["freeze_return_count"],
    )
    (OUT / "frozen_juejin_candidates.json").write_text(
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

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        definition = stage2_defs[str(item.case_id)]
        state = state_with_health(arrays, p, definition)
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        _, actions = run_case(arrays, p, state, exit_profile, record_actions=True)
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
