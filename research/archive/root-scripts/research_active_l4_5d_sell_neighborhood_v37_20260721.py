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
import research_active_l4_candidate_strength_v30_20260721 as v30
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_independent_sell_v36_20260721 as v36
import research_active_l4_market_state_v26_20260721 as market
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_5d_sell_neighborhood_v37_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


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
    g, rows = p["grid"], []
    for source in g["exit_score_source"]:
        for min_hold in g["min_hold"]:
            for max_hold in g["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_rank in g["sell_rank_below"]:
                    for advantage in g["replacement_advantage"]:
                        profile = v36.ExitProfile(source, min_hold, max_hold, sell_rank, advantage)
                        daily = v36.simulate(arrays, masked, buy_score, order, exposure, p["fixed_buy"], profile, p["observation_end"])
                        metrics = v30.evaluate(daily, p["year_folds"])
                        rows.append({"case_id": profile.case_id, **profile.__dict__, **metrics})
    results = pd.DataFrame(rows)
    results["eligible"] = (results.full_max_drawdown <= 0.40) & (results.full_trades >= 80)
    results = results.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    results.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = v30.dual_track(results, int(g["promote_return"]), int(g["promote_sharpe"]))
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        profile = v36.ExitProfile(**{key: item[key] for key in v36.ExitProfile.__dataclass_fields__})
        _, actions = v36.simulate(arrays, masked, buy_score, order, exposure, p["fixed_buy"], profile, p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "grid_cases": len(results), "all_year_positive": int(results.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "5D独立卖出邻域研究结论.md").write_text("\n".join(["# 5D独立卖出邻域研究结论", "", "全部参数在运行前冻结，只使用2026年以前观察期。", "", f"固定网格 {len(results)} 组，逐年为正 {int(results.all_year_positive.sum())} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
