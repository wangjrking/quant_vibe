# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_agreement_v8_20260721 as agreement
from . import production_v260_active_l4_buy_coverage_v39_20260721 as v39
from . import production_v260_active_l4_candidate_strength_v30_20260721 as v30
from . import production_v260_active_l4_independent_sell_v36_20260721 as v36
from . import production_v260_active_l4_market_state_v26_20260721 as market
from . import production_v260_active_l4_weak_open_refine_v49_20260721 as v49
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_continuous_market_v53_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"
CURRENT_TOP_WEIGHT = 0.95


def slot_weights(count, scheme):
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [CURRENT_TOP_WEIGHT, 1.0 - CURRENT_TOP_WEIGHT]


def rolling_mean(values, lookback):
    result = np.full(len(values), np.nan, dtype=np.float32)
    minimum = max(3, lookback // 4)
    for t in range(len(values)):
        part = values[max(0, t - lookback + 1):t + 1]
        part = part[np.isfinite(part)]
        if len(part) >= minimum:
            result[t] = float(np.mean(part))
    return result


def continuous_exposure(mean_return, floor, low, high):
    if high <= low:
        raise ValueError("linear high must exceed low")
    scaled = np.clip((mean_return - low) / (high - low), 0.0, 1.0)
    result = floor + (1.0 - floor) * scaled
    result[~np.isfinite(result)] = floor
    return result.astype(np.float32)


def main():
    global CURRENT_TOP_WEIGHT
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
    ms = p["market_state"]
    mean_return = rolling_mean(market.market_open_return(arrays), int(ms["lookback"]))
    market_on = np.isfinite(mean_return) & (mean_return > float(ms["entry_threshold"]))
    dynamic_entry = np.where(market_on, float(ms["risk_on_entry_rank_min"]), float(ms["risk_off_entry_rank_min"])).astype(np.float32)
    masked = {key: value.copy() for key, value in base_masked.items()}
    masked["signal_clean"] &= buy_score >= dynamic_entry[:, None]
    exit_profile = v36.ExitProfile(fixed["exit_score_source"], fixed["min_hold"], fixed["max_hold"], fixed["sell_rank_below"], fixed["replacement_advantage"])
    original_weights = v39.slot_weights
    v39.slot_weights = slot_weights
    try:
        rows, g, exposures = [], p["grid"], {}
        for floor in g["exposure_floor"]:
            for low in g["linear_low_mean_return"]:
                for high in g["linear_high_mean_return"]:
                    if high <= low:
                        continue
                    exposure = continuous_exposure(mean_return, float(floor), float(low), float(high))
                    exposure_id = f"f{int(floor*100)}_l{int(low*10000)}_h{int(high*10000)}"
                    exposures[exposure_id] = exposure
                    for top_weight in g["top1_weight"]:
                        CURRENT_TOP_WEIGHT = float(top_weight)
                        bp = v39.BuyProfile(2, -1.0, fixed["amount_min"], fixed["mv_min"], fixed["normal_total_pct"], False, 0.99, 0.005, 1.0, "dynamic")
                        daily = v39.simulate(arrays, masked, buy_score, order, exposure, bp, exit_profile, p["observation_end"])
                        metrics = v30.evaluate(daily, p["year_folds"])
                        cid = f"{exposure_id}_w{int(top_weight*100)}"
                        rows.append({"case_id": cid, "exposure_id": exposure_id, "exposure_floor": floor, "linear_low_mean_return": low, "linear_high_mean_return": high, "top1_weight": top_weight, **metrics})
        results = pd.DataFrame(rows)
        results["eligible"] = (results.all_year_positive) & (results.full_max_drawdown <= 0.40) & (results.full_trades >= 80)
        results = results.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
        results.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")
        pool = results[results.eligible]
        by_return = pool.sort_values(["full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, True]).head(int(g["promote_return"]))
        by_sharpe = pool.sort_values(["full_sharpe", "full_cumulative_return", "case_id"], ascending=[False, False, True]).head(int(g["promote_sharpe"]))
        frozen = pd.concat([by_return, by_sharpe], ignore_index=True).drop_duplicates("case_id")
        (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
        action_dir = OUT / "juejin_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        action_rows, seen = [], set()
        for _, item in frozen.iterrows():
            CURRENT_TOP_WEIGHT = float(item.top1_weight)
            bp = v39.BuyProfile(2, -1.0, fixed["amount_min"], fixed["mv_min"], fixed["normal_total_pct"], False, 0.99, 0.005, 1.0, "dynamic")
            _, actions = v39.simulate(arrays, masked, buy_score, order, exposures[str(item.exposure_id)], bp, exit_profile, p["observation_end"], record_actions=True)
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
    summary = {"status": "research_only_observation", "grid_cases": len(results), "all_year_positive": int(results.all_year_positive.sum()), "eligible_cases": len(pool), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "连续市场暴露结论.md").write_text("\n".join(["# 连续市场暴露结论", "", "本轮保持收益达标选股结构，仅将二档市场暴露改为连续线性暴露。", "", f"固定网格 {len(results)} 组，逐年为正 {int(results.all_year_positive.sum())} 组，强过滤后 {len(pool)} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "2026验证期未打开，生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
