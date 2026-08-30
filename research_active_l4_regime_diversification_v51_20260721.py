# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_buy_coverage_v39_20260721 as v39
import research_active_l4_candidate_strength_v30_20260721 as v30
import research_active_l4_independent_sell_v36_20260721 as v36
import research_active_l4_market_state_v26_20260721 as market
import research_active_l4_weak_open_refine_v49_20260721 as v49
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_regime_diversification_v51_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def slot_weights(count, scheme):
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    if scheme == "equal":
        return [1.0 / count] * count
    top = 0.60 if scheme == "top60" else 0.80
    return [top] + [(1.0 - top) / (count - 1)] * (count - 1)


def robust_sort(frame):
    return frame.sort_values(["all_year_positive", "min_year_cumulative_return", "median_year_sharpe", "full_cumulative_return", "case_id"], ascending=[False, False, False, False, True])


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / p["input_cache"]["path"]
    if p["status"] != "frozen_research_only" or v30.digest(cache) != p["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    buy_score = core.blend_scores(arrays, p["buy_score"])
    order = np.argsort(-np.nan_to_num(buy_score, nan=-np.inf), axis=1).astype(np.int32)
    base_masked = agreement.masked_arrays(arrays, p["agreement"])
    fixed = p["fixed"]
    base_masked = v49.gated_mask(base_masked, arrays, fixed["min_open_gap"])
    market_return = market.market_open_return(arrays)
    ms = p["market_state"]
    state = market.market_exposure(market_return, int(ms["lookback"]), float(ms["mean_return_threshold"]), 0.0) >= 0.999
    exit_profile = v36.ExitProfile(fixed["exit_score_source"], fixed["min_hold"], fixed["max_hold"], fixed["sell_rank_below"], fixed["replacement_advantage"])
    original_weights = v39.slot_weights
    v39.slot_weights = slot_weights
    try:
        rows, g, contexts = [], p["grid"], {}
        for top_n in g["top_n"]:
            for scheme in g["weight_scheme"]:
                for total in g["normal_total_pct"]:
                    for floor in g["risk_off_floor"]:
                        exposure = np.where(state, 1.0, float(floor)).astype(np.float32)
                        for off_entry in g["risk_off_entry_rank_min"]:
                            for on_entry in g["risk_on_entry_rank_min"]:
                                dynamic_entry = np.where(state, float(on_entry), float(off_entry)).astype(np.float32)
                                masked = {key: value.copy() for key, value in base_masked.items()}
                                masked["signal_clean"] &= buy_score >= dynamic_entry[:, None]
                                bp = v39.BuyProfile(int(top_n), -1.0, fixed["amount_min"], fixed["mv_min"], float(total), False, 0.99, 0.005, 1.0, scheme)
                                daily = v39.simulate(arrays, masked, buy_score, order, exposure, bp, exit_profile, p["observation_end"])
                                metrics = v30.evaluate(daily, p["year_folds"])
                                cid = f"n{top_n}_{scheme}_p{int(total*100)}_f{int(floor*100)}_oe{int(off_entry*1000)}_ne{int(on_entry*1000)}"
                                rows.append({"case_id": cid, "top_n": top_n, "weight_scheme": scheme, "normal_total_pct": total, "risk_off_floor": floor, "risk_off_entry_rank_min": off_entry, "risk_on_entry_rank_min": on_entry, **metrics})
        results = robust_sort(pd.DataFrame(rows))
        results["eligible"] = (results.full_max_drawdown <= 0.40) & (results.full_trades >= 80)
        results.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")
        pool = results[(results.all_year_positive) & results.eligible]
        by_return = pool.sort_values(["full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, True]).head(int(g["promote_return"]))
        by_robust = robust_sort(pool).head(int(g["promote_robust"]))
        frozen = pd.concat([by_return, by_robust], ignore_index=True).drop_duplicates("case_id")
        (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
        action_dir = OUT / "juejin_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        action_rows, seen = [], set()
        for _, item in frozen.iterrows():
            exposure = np.where(state, 1.0, float(item.risk_off_floor)).astype(np.float32)
            dynamic_entry = np.where(state, float(item.risk_on_entry_rank_min), float(item.risk_off_entry_rank_min)).astype(np.float32)
            masked = {key: value.copy() for key, value in base_masked.items()}
            masked["signal_clean"] &= buy_score >= dynamic_entry[:, None]
            bp = v39.BuyProfile(int(item.top_n), -1.0, fixed["amount_min"], fixed["mv_min"], float(item.normal_total_pct), False, 0.99, 0.005, 1.0, str(item.weight_scheme))
            _, actions = v39.simulate(arrays, masked, buy_score, order, exposure, bp, exit_profile, p["observation_end"], record_actions=True)
            content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
            if content_hash in seen:
                continue
            seen.add(content_hash)
            path = action_dir / f"{item.case_id}.csv"
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": int(item.top_n)})
        (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        v39.slot_weights = original_weights
    summary = {"status": "research_only_observation", "grid_cases": len(results), "all_year_positive": int(results.all_year_positive.sum()), "eligible_cases": len(pool), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "市场状态分散持仓结论.md").write_text("\n".join(["# 市场状态分散持仓结论", "", "本轮固定20日市场状态，联合调整持仓数量、分散权重、总仓位和动态入场阈值。", "", f"固定网格 {len(results)} 组，逐年为正 {int(results.all_year_positive.sum())} 组，强过滤后 {len(pool)} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "2026验证期未打开，生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
