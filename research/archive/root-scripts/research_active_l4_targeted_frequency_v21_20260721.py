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
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_yearly_robust_v11_20260721 as yearly
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_targeted_frequency_v21_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "目标频率邻域研究结论.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_id(profile):
    raw = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"))
    return "tf_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluate(daily, folds):
    result = yearly.evaluate(daily, folds)
    result["min_year_cumulative_return"] = min(float(result[f"{fold['id']}_cumulative_return"]) for fold in folds)
    return result


def dual_track(frame, return_count, sharpe_count):
    eligible = frame[frame.eligible & frame.all_year_positive]
    by_return = eligible.sort_values(["full_cumulative_return", "full_sharpe", "min_year_cumulative_return", "case_id"], ascending=[False, False, False, True]).head(return_count).assign(promotion_track="return")
    by_sharpe = eligible.sort_values(["full_sharpe", "full_cumulative_return", "min_year_cumulative_return", "case_id"], ascending=[False, False, False, True]).head(sharpe_count).assign(promotion_track="sharpe")
    return pd.concat([by_return, by_sharpe], ignore_index=True).drop_duplicates("case_id")


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {rule["id"]: core.blend_scores(arrays, rule) for rule in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32) for key, score in scores.items()}
    masked = agreement.masked_arrays(arrays, protocol["agreement"])
    qualities = {key: candidate_health.candidate_quality(masked, score, orders[key], int(protocol["health_to_exposure"]["shadow_top_n"])) for key, score in scores.items()}
    exposures = {(key, lookback, floor, denom): continuous.exposure_series(qualities[key], lookback, floor, denom) for key in scores for lookback in protocol["health_to_exposure"]["rolling_lookback"] for floor in protocol["health_to_exposure"]["floor_fraction"] for denom in protocol["health_to_exposure"]["scale_denominator"]}
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for lookback in protocol["health_to_exposure"]["rolling_lookback"]:
            for floor in protocol["health_to_exposure"]["floor_fraction"]:
                for denom in protocol["health_to_exposure"]["scale_denominator"]:
                    for amount_min in protocol["stage1"]["amount_min"]:
                        for entry_rank_min in protocol["stage1"]["entry_rank_min"]:
                            profile = continuous.ExposureProfile(blend_id, amount_min, 200000, int(protocol["stage1"]["top_n"]), entry_rank_min, lookback, floor, denom, **fixed)
                            daily = continuous.simulate(masked, scores[blend_id], orders[blend_id], exposures[(blend_id, lookback, floor, denom)], profile, protocol["observation_end"])
                            rows.append({"case_id": case_id(profile), **asdict(profile), **evaluate(daily, protocol["year_folds"])})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    seeds = dual_track(stage1, int(protocol["stage1"]["promote_return"]), int(protocol["stage1"]["promote_sharpe"]))
    rows = []
    grid = protocol["stage2"]
    for _, seed in seeds.iterrows():
        for min_hold in grid["min_hold"]:
            for max_hold in grid["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_rank_below in grid["sell_rank_below"]:
                    for replacement_advantage in grid["replacement_advantage"]:
                        for invested_ratio in grid["invested_ratio"]:
                            profile = continuous.ExposureProfile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), int(seed.health_lookback), float(seed.floor_fraction), float(seed.scale_denominator), min_hold, max_hold, sell_rank_below, replacement_advantage, invested_ratio)
                            daily = continuous.simulate(masked, scores[profile.blend_id], orders[profile.blend_id], exposures[(profile.blend_id, profile.health_lookback, profile.floor_fraction, profile.scale_denominator)], profile, protocol["observation_end"])
                            rows.append({"case_id": case_id(profile), "seed_case_id": str(seed.case_id), **asdict(profile), **evaluate(daily, protocol["year_folds"])})
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
    stage2 = stage2.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = dual_track(stage2, int(grid["promote_return"]), int(grid["promote_sharpe"]))
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for item in payload["profiles"]:
        profile = continuous.ExposureProfile(**{key: item[key] for key in continuous.ExposureProfile.__dataclass_fields__})
        _, actions = continuous.simulate(masked, scores[profile.blend_id], orders[profile.blend_id], exposures[(profile.blend_id, profile.health_lookback, profile.floor_fraction, profile.scale_denominator)], profile, protocol["observation_end"], record_actions=True)
        action_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_hash in seen:
            continue
        seen.add(action_hash)
        path = ACTION_DIR / f"{item['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item["case_id"], "promotion_track": item["promotion_track"], "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "content_sha256": action_hash, "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "best_return": frozen.sort_values("full_cumulative_return", ascending=False).head(1).to_dict("records"), "best_sharpe": frozen.sort_values("full_sharpe", ascending=False).head(1).to_dict("records"), "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(["# 目标频率邻域研究结论", "", "- 参数选择仅使用 `20220606-20251231`。", "- 第一阶段与第二阶段均采用收益轨和Sharpe轨双轨晋级。", "- `2026` 未参与筛选。", "- 本地只作预筛，正式结果以掘金为准。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结候选 {len(frozen)} 组，独立交易路径 {len(action_rows)} 条。", "", "本轮为 research-only，未修改生产策略或正式信号。"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"best_return", "best_sharpe"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
