# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_agreement_health_v16_20260721 as health
import research_active_l4_buy_coverage_v39_20260721 as v39
import research_active_l4_candidate_strength_v30_20260721 as v30
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_independent_sell_v36_20260721 as v36
import research_active_l4_market_state_v26_20260721 as market
import research_active_l4_top2_ratio_v41_20260721 as v41
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_robust_agreement_v44_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"
RATIOS = {"ratio75": 0.75, "ratio85": 0.85, "ratio90": 0.90}


def ratio_weights(count, scheme):
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    top = RATIOS[scheme]
    return [top, 1.0 - top]


def profile(values):
    return v39.BuyProfile(
        2,
        float(values["entry_rank_min"]),
        int(values["amount_min"]),
        int(values["mv_min"]),
        float(values["normal_total_pct"]),
        False,
        0.99,
        0.005,
        1.0,
        str(values["weight_scheme"]),
    )


def agreement_spec(r10, r5, r3):
    return {
        "id": f"a{int(round(r10 * 1000)):03d}_{int(round(r5 * 100)):02d}_{int(round(r3 * 100)):02d}",
        "rank10_min": float(r10),
        "rank5_min": float(r5),
        "rank3_min": float(r3),
    }


def row_id(spec, buy_profile):
    return f"{spec['id']}_{buy_profile.case_id}"


def robust_sort(frame):
    return frame.sort_values(
        ["all_year_positive", "min_year_cumulative_return", "median_year_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, True],
    )


def eligible(frame):
    return frame[(frame.all_year_positive) & (frame.full_max_drawdown <= 0.40) & (frame.full_trades >= 80)]


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / p["input_cache"]["path"]
    if p["status"] != "frozen_research_only" or v30.digest(cache) != p["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    buy_score = core.blend_scores(arrays, p["buy_score"])
    order = np.argsort(-np.nan_to_num(buy_score, nan=-np.inf), axis=1).astype(np.int32)
    e = p["exposure"]
    exit_profile = v36.ExitProfile(**p["fixed_exit"])
    original_weights = v39.slot_weights
    v39.slot_weights = ratio_weights
    try:
        stage1_rows = []
        s1 = p["stage1"]
        fixed_profile = profile(s1["fixed"])
        for r10 in s1["rank10_min"]:
            for r5 in s1["rank5_min"]:
                for r3 in s1["rank3_min"]:
                    spec = agreement_spec(r10, r5, r3)
                    masked = agreement.masked_arrays(arrays, spec)
                    quality = health.candidate_quality(masked, buy_score, order, 3)
                    model_exp = continuous.exposure_series(quality, int(e["health_lookback"]), float(e["health_floor_fraction"]), float(e["health_scale_denominator"]))
                    market_exp = market.market_exposure(market.market_open_return(arrays), int(e["market_lookback"]), float(e["market_threshold"]), float(e["market_floor"]))
                    exposure = np.minimum(model_exp, market_exp).astype(np.float32)
                    daily = v39.simulate(arrays, masked, buy_score, order, exposure, fixed_profile, exit_profile, p["observation_end"])
                    metrics = v30.evaluate(daily, p["year_folds"])
                    stage1_rows.append({"case_id": row_id(spec, fixed_profile), **spec, **fixed_profile.__dict__, **metrics})
        stage1 = robust_sort(pd.DataFrame(stage1_rows))
        stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
        stage1.to_csv(OUT / "stage1.csv", index=False, encoding="utf-8-sig")
        seeds = robust_sort(eligible(stage1)).head(int(s1["promote_robust"]))

        s2 = p["stage2"]
        stage2_rows = []
        for _, seed in seeds.iterrows():
            spec = agreement_spec(seed.rank10_min, seed.rank5_min, seed.rank3_min)
            masked = agreement.masked_arrays(arrays, spec)
            quality = health.candidate_quality(masked, buy_score, order, 3)
            model_exp = continuous.exposure_series(quality, int(e["health_lookback"]), float(e["health_floor_fraction"]), float(e["health_scale_denominator"]))
            market_exp = market.market_exposure(market.market_open_return(arrays), int(e["market_lookback"]), float(e["market_threshold"]), float(e["market_floor"]))
            exposure = np.minimum(model_exp, market_exp).astype(np.float32)
            for entry in s2["entry_rank_min"]:
                for amount in s2["amount_min"]:
                    for mv in s2["mv_min"]:
                        for total in s2["normal_total_pct"]:
                            for scheme in s2["weight_scheme"]:
                                bp = profile({"entry_rank_min": entry, "amount_min": amount, "mv_min": mv, "normal_total_pct": total, "weight_scheme": scheme})
                                daily = v39.simulate(arrays, masked, buy_score, order, exposure, bp, exit_profile, p["observation_end"])
                                metrics = v30.evaluate(daily, p["year_folds"])
                                stage2_rows.append({"case_id": row_id(spec, bp), **spec, **bp.__dict__, **metrics})
        stage2 = robust_sort(pd.DataFrame(stage2_rows).drop_duplicates("case_id"))
        stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
        stage2.to_csv(OUT / "stage2.csv", index=False, encoding="utf-8-sig")
        pool = eligible(stage2)
        by_return = pool.sort_values(["full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, True]).head(int(s2["promote_return"]))
        by_robust = robust_sort(pool).head(int(s2["promote_robust"]))
        frozen = pd.concat([by_return, by_robust], ignore_index=True).drop_duplicates("case_id")
        (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
        action_dir = OUT / "juejin_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        action_rows, seen = [], set()
        for _, item in frozen.iterrows():
            spec = agreement_spec(item.rank10_min, item.rank5_min, item.rank3_min)
            masked = agreement.masked_arrays(arrays, spec)
            quality = health.candidate_quality(masked, buy_score, order, 3)
            model_exp = continuous.exposure_series(quality, int(e["health_lookback"]), float(e["health_floor_fraction"]), float(e["health_scale_denominator"]))
            market_exp = market.market_exposure(market.market_open_return(arrays), int(e["market_lookback"]), float(e["market_threshold"]), float(e["market_floor"]))
            exposure = np.minimum(model_exp, market_exp).astype(np.float32)
            bp = v39.BuyProfile(**{key: item[key] for key in v39.BuyProfile.__dataclass_fields__})
            _, actions = v39.simulate(arrays, masked, buy_score, order, exposure, bp, exit_profile, p["observation_end"], record_actions=True)
            content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
            if content_hash in seen:
                continue
            seen.add(content_hash)
            path = action_dir / f"{item.case_id}.csv"
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
        (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        v39.slot_weights = original_weights
    summary = {
        "status": "research_only_observation",
        "stage1_cases": len(stage1),
        "stage1_all_year_positive": int(stage1.all_year_positive.sum()),
        "stage2_cases": len(stage2),
        "stage2_all_year_positive": int(stage2.all_year_positive.sum()),
        "frozen_candidates": len(frozen),
        "unique_juejin_paths": len(action_rows),
        "known_2026_used": False,
        "production_changed": False,
    }
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "稳健一致性研究结论.md").write_text("\n".join([
        "# 稳健一致性研究结论",
        "",
        "本轮在固定观察期内先优化多周期评分一致性，再优化Top2仓位与流动性。",
        "",
        f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。",
        f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。",
        f"冻结 {len(action_rows)} 条独立掘金路径。",
        "",
        "2026验证期未打开，生产资产未修改。",
    ]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
