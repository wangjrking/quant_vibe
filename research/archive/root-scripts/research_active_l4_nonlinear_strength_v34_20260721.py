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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_nonlinear_strength_v34_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def score_formula(arrays, spec):
    r3 = arrays["rank_3d"].astype(np.float64)
    r5 = arrays["rank_5d"].astype(np.float64)
    r10 = arrays["rank_10d"].astype(np.float64)
    linear = 0.70 * r10 + 0.20 * r5 + 0.10 * r3
    minimum = np.minimum(np.minimum(r3, r5), r10)
    kind, alpha = spec["kind"], float(spec["alpha"])
    if kind == "linear":
        value = linear
    elif kind == "min_mix":
        value = (1.0 - alpha) * linear + alpha * minimum
    elif kind == "geometric":
        value = np.power(np.maximum(r10, 1e-6), 0.70) * np.power(np.maximum(r5, 1e-6), 0.20) * np.power(np.maximum(r3, 1e-6), 0.10)
    elif kind == "harmonic":
        value = 1.0 / (0.70 / np.maximum(r10, 1e-6) + 0.20 / np.maximum(r5, 1e-6) + 0.10 / np.maximum(r3, 1e-6))
    elif kind == "mean_min":
        value = alpha * linear + (1.0 - alpha) * minimum
    else:
        raise ValueError(kind)
    return value.astype(np.float32)


def make_profile(fixed, entry, top, gap, exit_rule=None):
    values = dict(fixed)
    values.update({"entry_rank_min": entry, "strong_top_score_min": top, "strong_gap_min": gap})
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
    masked = agreement.masked_arrays(arrays, p["agreement"])
    e = p["exposure"]
    market_exp = market.market_exposure(market.market_open_return(arrays), int(e["market_lookback"]), float(e["market_threshold"]), float(e["market_floor"]))
    contexts, rows = {}, []
    g = p["stage1"]
    for spec in p["score_formulas"]:
        score = score_formula(arrays, spec)
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
        quality = health.candidate_quality(masked, score, order, 3)
        model_exp = continuous.exposure_series(quality, int(e["health_lookback"]), float(e["health_floor_fraction"]), float(e["health_scale_denominator"]))
        exposure = np.minimum(market_exp, model_exp).astype(np.float32)
        contexts[spec["id"]] = (score, order, exposure)
        for entry in g["entry_rank_min"]:
            for top in g["strong_top_score_min"]:
                for gap in g["strong_gap_min"]:
                    profile = make_profile(g["fixed"], entry, top, gap)
                    daily = v30.simulate(arrays, masked, score, order, exposure, profile, p["observation_end"])
                    metrics = v30.evaluate(daily, p["year_folds"])
                    cid = spec["id"] + "_" + v30.case_id(profile)
                    rows.append({"case_id": cid, "formula_id": spec["id"], **asdict(profile), **metrics})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage1.to_csv(OUT / "stage1.csv", index=False, encoding="utf-8-sig")
    seeds = v30.dual_track(stage1, int(g["promote_return"]), int(g["promote_sharpe"]))
    rows = []
    for _, seed in seeds.iterrows():
        score, order, exposure = contexts[str(seed.formula_id)]
        for exit_rule in p["stage2_exit_profiles"]:
            profile = make_profile(g["fixed"], float(seed.entry_rank_min), float(seed.strong_top_score_min), float(seed.strong_gap_min), exit_rule)
            daily = v30.simulate(arrays, masked, score, order, exposure, profile, p["observation_end"])
            metrics = v30.evaluate(daily, p["year_folds"])
            cid = str(seed.formula_id) + "_" + v30.case_id(profile)
            rows.append({"case_id": cid, "formula_id": seed.formula_id, "exit_profile_id": exit_rule["id"], **asdict(profile), **metrics})
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
        score, order, exposure = contexts[str(item.formula_id)]
        profile = v30.StrengthProfile(**{key: item[key] for key in v30.StrengthProfile.__dataclass_fields__})
        _, actions = v30.simulate(arrays, masked, score, order, exposure, profile, p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "formula_id": item.formula_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "非线性一致性评分研究结论.md").write_text("\n".join(["# 非线性一致性评分研究结论", "", "本轮评分公式与网格均在运行前冻结，只使用2026年以前观察期。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
