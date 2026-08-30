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
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_concentration_exit_v42_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def weights_for(count, scheme):
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    top = float(str(scheme).replace("ratio", "")) / 1000.0
    return [top, 1.0 - top]


def buy_profile(fixed, total, top_weight, strong):
    scheme = f"ratio{int(round(float(top_weight) * 1000))}"
    return v39.BuyProfile(2, float(fixed["entry_rank_min"]), int(fixed["amount_min"]), int(fixed["mv_min"]), float(total), bool(strong), float(fixed["strong_top_score_min"]), float(fixed["strong_gap_min"]), float(fixed["strong_target_pct"]), scheme)


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / p["input_cache"]["path"]
    if p["status"] != "frozen_research_only" or v30.digest(cache) != p["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    buy_score = core.blend_scores(arrays, p["buy_score"])
    order = np.argsort(-np.nan_to_num(buy_score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement.masked_arrays(arrays, p["agreement"])
    e = p["exposure"]
    quality = health.candidate_quality(masked, buy_score, order, 3)
    model_exp = continuous.exposure_series(quality, int(e["health_lookback"]), float(e["health_floor_fraction"]), float(e["health_scale_denominator"]))
    market_exp = market.market_exposure(market.market_open_return(arrays), int(e["market_lookback"]), float(e["market_threshold"]), float(e["market_floor"]))
    exposure = np.minimum(model_exp, market_exp).astype(np.float32)
    original = v39.slot_weights
    v39.slot_weights = weights_for
    try:
        s1, rows = p["stage1"], []
        fixed_exit = v36.ExitProfile(**s1["fixed_exit"])
        for total in s1["normal_total_pct"]:
            for top_weight in s1["top_weight"]:
                for strong in s1["strong_single_enabled"]:
                    bp = buy_profile(p["fixed_buy"], total, top_weight, strong)
                    daily = v39.simulate(arrays, masked, buy_score, order, exposure, bp, fixed_exit, p["observation_end"])
                    metrics = v30.evaluate(daily, p["year_folds"])
                    rows.append({"case_id": bp.case_id, "top_weight": top_weight, **bp.__dict__, **metrics})
        stage1 = pd.DataFrame(rows)
        stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
        stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
        stage1.to_csv(OUT / "stage1.csv", index=False, encoding="utf-8-sig")
        seeds = v30.dual_track(stage1, int(s1["promote_return"]), int(s1["promote_sharpe"]))
        s2, rows = p["stage2"], []
        for _, seed in seeds.iterrows():
            bp = buy_profile(p["fixed_buy"], seed.normal_total_pct, seed.top_weight, seed.strong_single_enabled)
            for min_hold in s2["min_hold"]:
                for max_hold in s2["max_hold"]:
                    for sell_rank in s2["sell_rank_below"]:
                        for advantage in s2["replacement_advantage"]:
                            ep = v36.ExitProfile("rank_5d", min_hold, max_hold, sell_rank, advantage)
                            daily = v39.simulate(arrays, masked, buy_score, order, exposure, bp, ep, p["observation_end"])
                            metrics = v30.evaluate(daily, p["year_folds"])
                            raw = json.dumps({"buy": bp.case_id, "exit": ep.__dict__}, sort_keys=True, separators=(",", ":"))
                            cid = "ce_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
                            rows.append({"case_id": cid, "buy_case_id": bp.case_id, "top_weight": seed.top_weight, **bp.__dict__, **ep.__dict__, **metrics})
        stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
        stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
        stage2 = stage2.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
        stage2.to_csv(OUT / "stage2.csv", index=False, encoding="utf-8-sig")
        frozen = v30.dual_track(stage2, int(s2["promote_return"]), int(s2["promote_sharpe"]))
        (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
        action_dir = OUT / "juejin_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        action_rows, seen = [], set()
        for _, item in frozen.iterrows():
            bp = v39.BuyProfile(**{key: item[key] for key in v39.BuyProfile.__dataclass_fields__})
            ep = v36.ExitProfile(**{key: item[key] for key in v36.ExitProfile.__dataclass_fields__})
            _, actions = v39.simulate(arrays, masked, buy_score, order, exposure, bp, ep, p["observation_end"], record_actions=True)
            content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
            if content_hash in seen:
                continue
            seen.add(content_hash)
            path = action_dir / f"{item.case_id}.csv"
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
        (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        v39.slot_weights = original
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "集中度与卖出邻域研究结论.md").write_text("\n".join(["# 集中度与卖出邻域研究结论", "", "本轮先细化Top2集中度，再对晋级仓位检验5D卖出邻域。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "2026验证期未打开，生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
