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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_score_rank_exit_v64_20260721"
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
    base_exit = sell_lib.ExitProfile(**p["stage1_fixed_exit"])
    stage1_rows, stage1_defs = [], {}
    for weights in p["stage1_grid"]["score_weights"]:
        for rank10_min in p["stage1_grid"]["rank10_min"]:
            for rank5_min in p["stage1_grid"]["rank5_min"]:
                for rank3_min in p["stage1_grid"]["rank3_min"]:
                    agreement = {
                        "id": "grid",
                        "rank10_min": rank10_min,
                        "rank5_min": rank5_min,
                        "rank3_min": rank3_min,
                    }
                    for entry in p["stage1_grid"]["entry_rank_pairs"]:
                        state = build_state(arrays, fixed, weights, agreement, entry)
                        daily = run_case(arrays, p, state, base_exit)
                        values = robust.evaluate_robust(daily, p)
                        definition = {
                            "weights": weights,
                            "agreement": agreement,
                            "entry": entry,
                            "exit": asdict(base_exit),
                        }
                        case_id = "sr1_" + stable_id(definition)
                        stage1_defs[case_id] = definition
                        stage1_rows.append({
                            "case_id": case_id,
                            "weights_json": json.dumps(weights, sort_keys=True),
                            "rank10_min": rank10_min,
                            "rank5_min": rank5_min,
                            "rank3_min": rank3_min,
                            "entry_json": json.dumps(entry, sort_keys=True),
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
        state = build_state(
            arrays,
            fixed,
            seed_def["weights"],
            seed_def["agreement"],
            seed_def["entry"],
        )
        for min_hold in p["stage2_grid"]["min_hold"]:
            for max_hold in p["stage2_grid"]["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_rank_below in p["stage2_grid"]["sell_rank_below"]:
                    for replacement_advantage in p["stage2_grid"]["replacement_advantage"]:
                        exit_profile = sell_lib.ExitProfile(
                            "rank_5d",
                            min_hold,
                            max_hold,
                            sell_rank_below,
                            replacement_advantage,
                        )
                        daily = run_case(arrays, p, state, exit_profile)
                        values = robust.evaluate_robust(daily, p)
                        definition = {
                            "seed_case_id": str(seed.case_id),
                            "weights": seed_def["weights"],
                            "agreement": seed_def["agreement"],
                            "entry": seed_def["entry"],
                            "exit": asdict(exit_profile),
                        }
                        case_id = "sr2_" + stable_id(definition)
                        stage2_defs[case_id] = definition
                        stage2_rows.append({
                            "case_id": case_id,
                            "seed_case_id": str(seed.case_id),
                            "weights_json": json.dumps(seed_def["weights"], sort_keys=True),
                            "rank10_min": seed_def["agreement"]["rank10_min"],
                            "rank5_min": seed_def["agreement"]["rank5_min"],
                            "rank3_min": seed_def["agreement"]["rank3_min"],
                            "entry_json": json.dumps(seed_def["entry"], sort_keys=True),
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
        state = build_state(
            arrays,
            fixed,
            definition["weights"],
            definition["agreement"],
            definition["entry"],
        )
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
