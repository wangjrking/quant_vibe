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
import research_active_l4_breadth_dynamic_v13_20260721 as dynamic
import research_active_l4_lagged_health_gate_v15_20260721 as health
import research_active_l4_yearly_robust_v11_20260721 as yearly
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_agreement_health_v16_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "一致性前排健康度研究结论.md"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_id(profile, agreement_id: str, lookback: int, shadow_top_n: int) -> str:
    raw = json.dumps({"profile": asdict(profile), "agreement_id": agreement_id, "lookback": lookback, "shadow_top_n": shadow_top_n}, sort_keys=True, separators=(",", ":"))
    return "agh_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def candidate_quality(arrays, score, order, top_n):
    opens = arrays["buy_open"]
    result = np.full(len(opens), np.nan, dtype=float)
    for s in range(len(opens) - 1):
        valid = arrays["signal_clean"][s] & np.isfinite(score[s]) & np.isfinite(opens[s]) & (opens[s] > 0) & np.isfinite(opens[s + 1]) & (opens[s + 1] > 0)
        chosen = [int(idx) for idx in order[s] if valid[int(idx)]][:top_n]
        if chosen:
            result[s] = float(np.mean(opens[s + 1, chosen] / opens[s, chosen] - 1.0) - 0.0066)
    return result


def evaluate(daily, folds):
    result = yearly.evaluate(daily, folds)
    result["min_year_cumulative_return"] = min(float(result[f"{fold['id']}_cumulative_return"]) for fold in folds)
    return result


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {item["id"]: core.blend_scores(arrays, item) for item in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    gates = {item["id"]: item for item in protocol["agreement_grid"]}
    masked = {key: agreement.masked_arrays(arrays, item) for key, item in gates.items()}
    states = {}
    for blend_id, score in scores.items():
        for agreement_id, masked_arrays in masked.items():
            for shadow_top_n in protocol["health_gate"]["shadow_top_n"]:
                quality = candidate_quality(masked_arrays, score, orders[blend_id], shadow_top_n)
                for lookback in protocol["health_gate"]["rolling_lookback"]:
                    states[(blend_id, agreement_id, shadow_top_n, lookback)] = health.health_state(quality, lookback)
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for agreement_id in gates:
            for shadow_top_n in protocol["health_gate"]["shadow_top_n"]:
                for lookback in protocol["health_gate"]["rolling_lookback"]:
                    for amount_min in protocol["stage1"]["amount_min"]:
                        for top_n in protocol["stage1"]["top_n"]:
                            for entry_rank_min in protocol["stage1"]["entry_rank_min"]:
                                profile = dynamic.DynamicProfile(blend_id, amount_min, 200000, top_n, entry_rank_min, lookback, float(shadow_top_n), **fixed)
                                daily = dynamic.simulate(masked[agreement_id], scores[blend_id], orders[blend_id], states[(blend_id, agreement_id, shadow_top_n, lookback)], profile, protocol["observation_end"])
                                rows.append({"case_id": case_id(profile, agreement_id, lookback, shadow_top_n), "agreement_id": agreement_id, "shadow_top_n": shadow_top_n, "gate_lookback": lookback, **asdict(profile), **evaluate(daily, protocol["year_folds"])})
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
                                profile = dynamic.DynamicProfile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), int(seed.gate_lookback), float(seed.shadow_top_n), min_hold, max_hold, sell_rank_below, replacement_advantage, invested_ratio, weak_fraction)
                                state = states[(profile.blend_id, str(seed.agreement_id), int(seed.shadow_top_n), profile.breadth_lookback)]
                                daily = dynamic.simulate(masked[str(seed.agreement_id)], scores[profile.blend_id], orders[profile.blend_id], state, profile, protocol["observation_end"])
                                rows.append({"case_id": case_id(profile, str(seed.agreement_id), profile.breadth_lookback, int(seed.shadow_top_n)), "seed_case_id": str(seed.case_id), "agreement_id": str(seed.agreement_id), "shadow_top_n": int(seed.shadow_top_n), "gate_lookback": profile.breadth_lookback, **asdict(profile), **evaluate(daily, protocol["year_folds"])})
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
        state = states[(profile.blend_id, item["agreement_id"], int(item["shadow_top_n"]), profile.breadth_lookback)]
        _, actions = dynamic.simulate(masked[item["agreement_id"]], scores[profile.blend_id], orders[profile.blend_id], state, profile, protocol["observation_end"], record_actions=True)
        action_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_hash in seen:
            continue
        seen.add(action_hash)
        path = ACTION_DIR / f"{item['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item["case_id"], "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    best = frozen.head(1).to_dict("records")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "best": best, "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    best_line = "无逐年为正候选。" if not best else f"最佳候选 `{best[0]['case_id']}`：本地观察期累计收益 {best[0]['full_cumulative_return']:.2%}，Sharpe {best[0]['full_sharpe']:.3f}，最大回撤 {best[0]['full_max_drawdown']:.2%}。"
    REPORT_PATH.write_text("\n".join(["# 一致性前排健康度研究结论", "", "- 参数选择仅使用 `20220606-20251231`。", "- 健康度严格滞后且扣除往返成本。", "- `2026` 未参与筛选。", "- 本地只作预筛，正式结果以掘金为准。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", best_line, "", "本轮为 research-only，未修改生产策略或正式信号。"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
