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
import research_active_l4_liquidity_risk_v59_20260721 as liquidity
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_market_trend_exposure_v63_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def market_trend(close_qfq, lookback):
    result = np.full(close_qfq.shape[0], np.nan, dtype=np.float32)
    for t in range(lookback, close_qfq.shape[0]):
        current = close_qfq[t]
        previous = close_qfq[t - lookback]
        valid = (
            np.isfinite(current)
            & (current > 0)
            & np.isfinite(previous)
            & (previous > 0)
        )
        if valid.any():
            result[t] = float(np.median(current[valid] / previous[valid] - 1.0))
    return result


def continuous_exposure(trend, floor, lower, upper):
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    scaled = (trend - lower) / (upper - lower)
    scaled = np.clip(scaled, 0.0, 1.0)
    exposure = floor + (1.0 - floor) * scaled
    exposure[~np.isfinite(trend)] = floor
    return exposure.astype(np.float32)


def apply_filter(state, arrays, turnover_max):
    score, order, masked, exposure = state
    masked = {key: value.copy() for key, value in masked.items()}
    valid = np.isfinite(arrays["turnover_rate"])
    valid &= arrays["turnover_rate"] >= 0.0
    valid &= arrays["turnover_rate"] <= turnover_max
    valid &= arrays["amount"] >= 90000
    valid &= arrays["total_mv"] >= 200000
    masked["signal_clean"] &= valid
    return score, order, masked, exposure


def run_case(arrays, protocol, state, exit_profile, record_actions=False):
    buy = simulator.BuyProfile(
        2, -1.0, 90000, 200000, 1.0, False, 0.99, 0.005, 1.0, "equal"
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

    fixed = p["fixed"]
    base_state = robust.build_state(
        arrays,
        fixed["score_weights"],
        fixed["agreement"],
        fixed["entry"]["risk_off"],
        fixed["entry"]["risk_on"],
        fixed,
    )
    base_state = apply_filter(base_state, arrays, fixed["turnover_max"])
    trends = {
        lookback: market_trend(arrays["close_qfq"], lookback)
        for lookback in p["stage1_grid"]["lookback"]
    }

    stage1_rows, stage1_defs = [], {}
    for lookback, trend in trends.items():
        for floor in p["stage1_grid"]["floor"]:
            for band in p["stage1_grid"]["bands"]:
                exposure = continuous_exposure(trend, floor, band["lower"], band["upper"])
                state = (base_state[0], base_state[1], base_state[2], exposure)
                for exit_values in p["stage1_grid"]["exit_profiles"]:
                    exit_profile = sell_lib.ExitProfile("rank_5d", **exit_values)
                    daily = run_case(arrays, p, state, exit_profile)
                    values = robust.evaluate_robust(daily, p)
                    definition = {
                        "lookback": lookback,
                        "floor": floor,
                        "band": band,
                        "exit": asdict(exit_profile),
                    }
                    case_id = "mt1_" + stable_id(definition)
                    stage1_defs[case_id] = definition
                    stage1_rows.append({
                        "case_id": case_id,
                        "lookback": lookback,
                        "floor": floor,
                        "lower": band["lower"],
                        "upper": band["upper"],
                        **asdict(exit_profile),
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

    stage2_rows, stage2_defs, state_cache = [], {}, {}
    for _, seed in promoted.iterrows():
        seed_def = stage1_defs[str(seed.case_id)]
        exposure = continuous_exposure(
            trends[int(seed_def["lookback"])],
            seed_def["floor"],
            seed_def["band"]["lower"],
            seed_def["band"]["upper"],
        )
        for agreement in p["stage2_grid"]["agreement_profiles"]:
            for entry in p["stage2_grid"]["entry_rank_pairs"]:
                state_key = stable_id({
                    "seed": str(seed.case_id),
                    "agreement": agreement,
                    "entry": entry,
                })
                if state_key not in state_cache:
                    state = robust.build_state(
                        arrays,
                        fixed["score_weights"],
                        agreement,
                        entry["risk_off"],
                        entry["risk_on"],
                        fixed,
                    )
                    state = apply_filter(state, arrays, fixed["turnover_max"])
                    state_cache[state_key] = (state[0], state[1], state[2], exposure)
                state = state_cache[state_key]
                for sell_rank_below in p["stage2_grid"]["sell_rank_below"]:
                    for replacement_advantage in p["stage2_grid"]["replacement_advantage"]:
                        exit_profile = sell_lib.ExitProfile(
                            "rank_5d",
                            int(seed_def["exit"]["min_hold"]),
                            int(seed_def["exit"]["max_hold"]),
                            sell_rank_below,
                            replacement_advantage,
                        )
                        daily = run_case(arrays, p, state, exit_profile)
                        values = robust.evaluate_robust(daily, p)
                        definition = {
                            "seed_case_id": str(seed.case_id),
                            "lookback": seed_def["lookback"],
                            "floor": seed_def["floor"],
                            "band": seed_def["band"],
                            "agreement": agreement,
                            "entry": entry,
                            "exit": asdict(exit_profile),
                            "state_key": state_key,
                        }
                        case_id = "mt2_" + stable_id(definition)
                        stage2_defs[case_id] = definition
                        stage2_rows.append({
                            "case_id": case_id,
                            "seed_case_id": str(seed.case_id),
                            "lookback": seed_def["lookback"],
                            "floor": seed_def["floor"],
                            "lower": seed_def["band"]["lower"],
                            "upper": seed_def["band"]["upper"],
                            "agreement_json": json.dumps(agreement, sort_keys=True),
                            "entry_json": json.dumps(entry, sort_keys=True),
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
        state = state_cache[definition["state_key"]]
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
