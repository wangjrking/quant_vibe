# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_agreement_health_v16_20260721 as candidate_health
import research_active_l4_candidate_strength_v30_20260721 as v30
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_market_state_v26_20260721 as market
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_strength_neighborhood_v31_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def build_profile(values, exit_rule):
    return v30.StrengthProfile(
        int(values[0]), int(values[1]), float(values[2]), float(values[3]),
        float(values[4]), float(values[5]), float(values[6]),
        int(exit_rule["min_hold"]), int(exit_rule["max_hold"]),
        float(exit_rule["sell_rank_below"]), float(exit_rule["replacement_advantage"]),
    )


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or v30.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score = core.blend_scores(arrays, protocol["score"])
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement.masked_arrays(arrays, protocol["agreement"])
    h = protocol["model_health"]
    quality = candidate_health.candidate_quality(masked, score, order, int(h["shadow_top_n"]))
    model_exp = continuous.exposure_series(quality, int(h["rolling_lookback"]), float(h["floor_fraction"]), float(h["scale_denominator"]))
    m = protocol["market_state"]
    market_exp = market.market_exposure(market.market_open_return(arrays), int(m["lookback"]), float(m["mean_return_threshold"]), float(m["risk_off_floor"]))
    exposure = np.minimum(model_exp, market_exp).astype(np.float32)
    grid = protocol["stage1"]
    fixed = grid["fixed"]
    fixed_exit = {"min_hold": fixed["min_hold"], "max_hold": fixed["max_hold"], "sell_rank_below": fixed["sell_rank_below"], "replacement_advantage": fixed["replacement_advantage"]}
    rows = []
    for amount in grid["amount_min"]:
        for mv in grid["total_mv_min"]:
            for entry in grid["entry_rank_min"]:
                for top in grid["strong_top_score_min"]:
                    for gap in grid["strong_gap_min"]:
                        for strong_pct in grid["strong_target_pct"]:
                            for normal_pct in grid["normal_total_pct"]:
                                profile = build_profile((amount, mv, entry, top, gap, strong_pct, normal_pct), fixed_exit)
                                daily = v30.simulate(arrays, masked, score, order, exposure, profile, protocol["observation_end"])
                                metrics = v30.evaluate(daily, protocol["year_folds"])
                                metrics["strong_days"] = int(daily["strong_mode"].sum())
                                rows.append({"case_id": v30.case_id(profile), **asdict(profile), **metrics})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage1.to_csv(OUT / "stage1.csv", index=False, encoding="utf-8-sig")
    seeds = v30.dual_track(stage1, int(grid["promote_return"]), int(grid["promote_sharpe"]))
    rows = []
    for _, seed in seeds.iterrows():
        base = (seed.amount_min, seed.mv_min, seed.entry_rank_min, seed.strong_top_score_min, seed.strong_gap_min, seed.strong_target_pct, seed.normal_total_pct)
        for exit_rule in protocol["stage2_exit_profiles"]:
            profile = build_profile(base, exit_rule)
            daily = v30.simulate(arrays, masked, score, order, exposure, profile, protocol["observation_end"])
            metrics = v30.evaluate(daily, protocol["year_folds"])
            metrics["strong_days"] = int(daily["strong_mode"].sum())
            rows.append({"case_id": v30.case_id(profile), "exit_profile_id": exit_rule["id"], **asdict(profile), **metrics})
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
    stage2 = stage2.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage2.to_csv(OUT / "stage2.csv", index=False, encoding="utf-8-sig")
    frozen = v30.dual_track(stage2, int(protocol["stage2"]["promote_return"]), int(protocol["stage2"]["promote_sharpe"]))
    payload = {"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    actions_out, seen = [], set()
    for item in payload["profiles"]:
        profile = v30.StrengthProfile(**{key: item[key] for key in v30.StrengthProfile.__dataclass_fields__})
        _, actions = v30.simulate(arrays, masked, score, order, exposure, profile, protocol["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        actions_out.append({"case_id": item["case_id"], "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": actions_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(actions_out), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "强信号邻域研究结论.md").write_text("\n".join(["# 强信号邻域研究结论", "", "本轮仅使用2026年以前观察期，并在运行前冻结全部阈值。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结 {len(actions_out)} 条独立掘金路径。", "", "本轮未读取2026数据，未修改生产策略。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
