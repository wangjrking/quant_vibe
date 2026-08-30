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
import research_active_l4_agreement_health_v16_20260721 as health
import research_active_l4_candidate_strength_v30_20260721 as v30
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_market_state_v26_20260721 as market
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_exposure_strength_v32_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def exposure_id(params):
    raw = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return "ex_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_profile(base, exit_rule=None):
    values = dict(base)
    if exit_rule:
        values.update({key: exit_rule[key] for key in ("min_hold", "max_hold", "sell_rank_below", "replacement_advantage")})
    return v30.StrengthProfile(**values)


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / p["input_cache"]["path"]
    if p["status"] != "frozen_research_only" or v30.digest(cache) != p["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score = core.blend_scores(arrays, p["score"])
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement.masked_arrays(arrays, p["agreement"])
    quality = health.candidate_quality(masked, score, order, 3)
    market_return = market.market_open_return(arrays)
    profile = make_profile(p["fixed_profile"])
    g = p["stage1"]
    rows, exposure_cache = [], {}
    for ml in g["market_lookback"]:
        for mt in g["market_threshold"]:
            for mf in g["market_floor"]:
                market_exp = market.market_exposure(market_return, int(ml), float(mt), float(mf))
                for hl in g["health_lookback"]:
                    for hf in g["health_floor_fraction"]:
                        for hd in g["health_scale_denominator"]:
                            model_exp = continuous.exposure_series(quality, int(hl), float(hf), float(hd))
                            exposure = np.minimum(market_exp, model_exp).astype(np.float32)
                            params = {"market_lookback": ml, "market_threshold": mt, "market_floor": mf, "health_lookback": hl, "health_floor_fraction": hf, "health_scale_denominator": hd}
                            eid = exposure_id(params)
                            exposure_cache[eid] = exposure
                            daily = v30.simulate(arrays, masked, score, order, exposure, profile, p["observation_end"])
                            metrics = v30.evaluate(daily, p["year_folds"])
                            rows.append({"exposure_id": eid, **params, **metrics})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "exposure_id"], ascending=[False, False, False, True])
    stage1.to_csv(OUT / "stage1.csv", index=False, encoding="utf-8-sig")
    seeds = v30.dual_track(stage1.rename(columns={"exposure_id": "case_id"}), int(g["promote_return"]), int(g["promote_sharpe"])).rename(columns={"case_id": "exposure_id"})
    rows = []
    for _, seed in seeds.iterrows():
        params = {key: seed[key] for key in ("market_lookback", "market_threshold", "market_floor", "health_lookback", "health_floor_fraction", "health_scale_denominator")}
        exposure = exposure_cache[str(seed.exposure_id)]
        for exit_rule in p["stage2_exit_profiles"]:
            tuned = make_profile(p["fixed_profile"], exit_rule)
            daily = v30.simulate(arrays, masked, score, order, exposure, tuned, p["observation_end"])
            metrics = v30.evaluate(daily, p["year_folds"])
            cid = exposure_id({**params, "exit": exit_rule["id"]})
            rows.append({"case_id": cid, "exposure_id": seed.exposure_id, "exit_profile_id": exit_rule["id"], **params, **asdict(tuned), **metrics})
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
    stage2 = stage2.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage2.to_csv(OUT / "stage2.csv", index=False, encoding="utf-8-sig")
    frozen = v30.dual_track(stage2, int(p["stage2"]["promote_return"]), int(p["stage2"]["promote_sharpe"]))
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        tuned = v30.StrengthProfile(**{key: item[key] for key in v30.StrengthProfile.__dataclass_fields__})
        exposure = exposure_cache[str(item.exposure_id)]
        _, actions = v30.simulate(arrays, masked, score, order, exposure, tuned, p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "暴露强度联合研究结论.md").write_text("\n".join(["# 暴露强度联合研究结论", "", "本轮固定V30选股结构，只优化市场状态与模型健康暴露。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "2026数据未参与筛选，生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
