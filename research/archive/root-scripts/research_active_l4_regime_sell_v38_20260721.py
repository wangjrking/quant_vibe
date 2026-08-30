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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_regime_sell_v38_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def source_array(arrays, buy_score, source):
    return v36.exit_score_array(arrays, buy_score, source)


def case_id(values):
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return "rs_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
    risk_on = market_exp >= 0.999
    sources = {name: source_array(arrays, buy_score, name) for name in p["grid"]["risk_on_source"]}
    g, rows, contexts = p["grid"], [], {}
    for on_source in g["risk_on_source"]:
        for off_source in g["risk_off_source"]:
            dynamic = np.where(risk_on[:, None], sources[on_source], sources[off_source]).astype(np.float32)
            context_id = f"{on_source}__{off_source}"
            dynamic_arrays = dict(arrays)
            dynamic_arrays["rank_5d"] = dynamic
            contexts[context_id] = dynamic_arrays
            for min_hold in g["min_hold"]:
                for max_hold in g["max_hold"]:
                    for sell_rank in g["sell_rank_below"]:
                        for advantage in g["replacement_advantage"]:
                            values = {"risk_on_source": on_source, "risk_off_source": off_source, "min_hold": min_hold, "max_hold": max_hold, "sell_rank_below": sell_rank, "replacement_advantage": advantage}
                            cid = case_id(values)
                            profile = v36.ExitProfile("rank_5d", min_hold, max_hold, sell_rank, advantage)
                            daily = v36.simulate(dynamic_arrays, masked, buy_score, order, exposure, p["fixed_buy"], profile, p["observation_end"])
                            metrics = v30.evaluate(daily, p["year_folds"])
                            rows.append({"case_id": cid, "context_id": context_id, **values, **metrics})
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
        profile = v36.ExitProfile("rank_5d", int(item.min_hold), int(item.max_hold), float(item.sell_rank_below), float(item.replacement_advantage))
        _, actions = v36.simulate(contexts[str(item.context_id)], masked, buy_score, order, exposure, p["fixed_buy"], profile, p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "risk_on_source": item.risk_on_source, "risk_off_source": item.risk_off_source, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "grid_cases": len(results), "all_year_positive": int(results.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "市场状态卖出研究结论.md").write_text("\n".join(["# 市场状态卖出研究结论", "", "风险状态只使用T日及更早市场数据，参数运行前冻结。", "", f"固定网格 {len(results)} 组，逐年为正 {int(results.all_year_positive.sum())} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "2026验证期未打开，生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
