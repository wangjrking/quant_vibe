# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_breadth_dynamic_v13_20260721 as dynamic
import research_active_l4_mature_horizon_v14_20260721 as horizon
import research_active_l4_yearly_robust_v11_20260721 as yearly
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_lagged_health_gate_v15_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "滞后模型健康度风控研究结论.md"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shadow_quality(arrays: dict[str, np.ndarray], score: np.ndarray) -> np.ndarray:
    opens = arrays["buy_open"]
    result = np.full(len(opens), np.nan, dtype=float)
    for s in range(len(opens) - 1):
        valid = arrays["signal_clean"][s] & np.isfinite(score[s]) & (score[s] >= 0.90) & np.isfinite(opens[s]) & (opens[s] > 0) & np.isfinite(opens[s + 1]) & (opens[s + 1] > 0)
        if valid.any():
            result[s] = float(np.mean(opens[s + 1, valid] / opens[s, valid] - 1.0) - 0.0066)
    return result


def health_state(quality: np.ndarray, lookback: int) -> np.ndarray:
    result = np.zeros(len(quality), dtype=bool)
    minimum = max(5, lookback // 4)
    for t in range(len(quality)):
        end = t - 1
        start = max(0, end - lookback)
        values = quality[start:end]
        values = values[np.isfinite(values)]
        if len(values) >= minimum:
            result[t] = float(np.mean(values)) > 0.0
    return result


def evaluate(daily: pd.DataFrame, folds: list[dict]) -> dict:
    result = yearly.evaluate(daily, folds)
    year_returns = [float(result[f"{fold['id']}_cumulative_return"]) for fold in folds]
    result["min_year_cumulative_return"] = min(year_returns)
    return result


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {}
    for rule in protocol["score_rules"]:
        if rule["type"] == "fixed":
            scores[rule["id"]] = core.blend_scores(arrays, rule)
        else:
            scores[rule["id"]], _ = horizon.dynamic_score(arrays, int(rule["quality_lookback"]), [3, 5, 10], 10)
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    states = {(key, lookback): health_state(shadow_quality(arrays, value), lookback) for key, value in scores.items() for lookback in protocol["health_gate"]["rolling_lookback"]}
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for gate_lookback in protocol["health_gate"]["rolling_lookback"]:
            for amount_min in protocol["stage1"]["amount_min"]:
                for mv_min in protocol["stage1"]["total_mv_min"]:
                    for top_n in protocol["stage1"]["top_n"]:
                        for entry_rank_min in protocol["stage1"]["entry_rank_min"]:
                            profile = dynamic.DynamicProfile(blend_id, amount_min, mv_min, top_n, entry_rank_min, gate_lookback, 0.0, **fixed)
                            daily = dynamic.simulate(arrays, scores[blend_id], orders[blend_id], states[(blend_id, gate_lookback)], profile, protocol["observation_end"])
                            rows.append({"case_id": profile.case_id, "gate_lookback": gate_lookback, **asdict(profile), **evaluate(daily, protocol["year_folds"])})
    stage1 = pd.DataFrame(rows)
    gate = protocol["eligibility"]
    stage1["eligible"] = (stage1.full_max_drawdown <= float(gate["full_max_drawdown_max"])) & (stage1.full_trades >= int(gate["full_trades_min"]))
    sort_columns = ["all_year_positive", "min_year_cumulative_return", "min_year_sharpe", "median_year_sharpe", "full_sharpe", "full_max_drawdown", "case_id"]
    ascending = [False, False, False, False, False, True, True]
    stage1 = stage1.sort_values(sort_columns, ascending=ascending)
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    seeds = stage1[stage1.eligible].head(int(protocol["stage1"]["promote"]))
    rows = []
    grid = protocol["stage2"]
    for _, seed in seeds.iterrows():
        for min_hold in grid["min_hold"]:
            for max_hold in grid["max_hold"]:
                for sell_rank_below in grid["sell_rank_below"]:
                    for replacement_advantage in grid["replacement_advantage"]:
                        for invested_ratio in grid["invested_ratio"]:
                            for weak_fraction in grid["weak_position_fraction"]:
                                profile = dynamic.DynamicProfile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), int(seed.gate_lookback), 0.0, min_hold, max_hold, sell_rank_below, replacement_advantage, invested_ratio, weak_fraction)
                                daily = dynamic.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], states[(profile.blend_id, profile.breadth_lookback)], profile, protocol["observation_end"])
                                rows.append({"case_id": profile.case_id, "seed_case_id": str(seed.case_id), "gate_lookback": profile.breadth_lookback, **asdict(profile), **evaluate(daily, protocol["year_folds"])})
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["eligible"] = (stage2.full_max_drawdown <= float(gate["full_max_drawdown_max"])) & (stage2.full_trades >= int(gate["full_trades_min"]))
    stage2 = stage2.sort_values(sort_columns, ascending=ascending)
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = stage2[stage2.eligible & stage2.all_year_positive].head(int(grid["promote_for_juejin"]))
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for item in payload["profiles"]:
        profile = dynamic.DynamicProfile(**{key: item[key] for key in dynamic.DynamicProfile.__dataclass_fields__})
        _, actions = dynamic.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], states[(profile.blend_id, profile.breadth_lookback)], profile, protocol["observation_end"], record_actions=True)
        action_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_hash in seen:
            continue
        seen.add(action_hash)
        path = ACTION_DIR / f"{profile.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": profile.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    health_counts = {f"{key}_lb{lookback}": int(value.sum()) for (key, lookback), value in states.items()}
    best = frozen.head(1).to_dict("records")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "healthy_day_counts": health_counts, "best": best, "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    best_line = "无逐年为正候选。" if not best else f"最佳候选 `{best[0]['case_id']}`：本地观察期累计收益 {best[0]['full_cumulative_return']:.2%}，Sharpe {best[0]['full_sharpe']:.3f}，最大回撤 {best[0]['full_max_drawdown']:.2%}。"
    REPORT_PATH.write_text("\n".join(["# 滞后模型健康度风控研究结论", "", "- 参数选择只使用 `20220606-20251231`。", "- 健康度只读取已经成熟且扣除成本的历史收益。", "- `2026` 未参与筛选。", "- 本地仅作预筛，正式收益以掘金为准。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", best_line, "", "本轮为 research-only，未修改生产策略或正式信号。"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
