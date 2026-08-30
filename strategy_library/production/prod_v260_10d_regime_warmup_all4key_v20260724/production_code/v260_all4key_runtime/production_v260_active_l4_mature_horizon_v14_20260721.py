# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_best_corrected_v6_20260721 as engine
from . import production_v260_active_l4_yearly_robust_v11_20260721 as yearly
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_mature_horizon_v14_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "成熟期限自适应研究结论.md"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality_observations(arrays: dict[str, np.ndarray], horizon: int) -> np.ndarray:
    rank = arrays[f"rank_{horizon}d"]
    opens = arrays["buy_open"]
    result = np.full(len(opens), np.nan, dtype=float)
    for s in range(len(opens) - horizon):
        valid = arrays["signal_clean"][s] & np.isfinite(rank[s]) & (rank[s] >= 0.90) & np.isfinite(opens[s]) & (opens[s] > 0) & np.isfinite(opens[s + horizon]) & (opens[s + horizon] > 0)
        if valid.any():
            result[s] = float(np.mean(opens[s + horizon, valid] / opens[s, valid] - 1.0))
    return result


def dynamic_score(arrays: dict[str, np.ndarray], lookback: int, horizons: list[int], default_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    qualities = {h: quality_observations(arrays, h) for h in horizons}
    ranks = {h: arrays[f"rank_{h}d"] for h in horizons}
    selected = np.full(len(arrays["dates"]), default_horizon, dtype=np.int16)
    score = np.full_like(arrays["rank_10d"], np.nan, dtype=np.float32)
    priority = {10: 2, 5: 1, 3: 0}
    for t in range(len(selected)):
        candidates = []
        for h in horizons:
            mature_end = t - h
            mature_start = max(0, mature_end - lookback)
            values = qualities[h][mature_start:mature_end]
            values = values[np.isfinite(values)]
            if len(values) >= max(5, lookback // 4):
                candidates.append((float(np.mean(values)), priority.get(h, 0), h))
        if candidates:
            selected[t] = max(candidates)[2]
        score[t] = ranks[int(selected[t])][t]
    return score, selected


def evaluate(daily: pd.DataFrame, folds: list[dict]) -> dict:
    result = yearly.evaluate(daily, folds)
    returns = [float(result[f"{fold['id']}_cumulative_return"]) for fold in folds]
    result["min_year_cumulative_return"] = min(returns)
    return result


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    selector = protocol["horizon_selector"]
    scores, selections, orders = {}, {}, {}
    for lookback in selector["rolling_lookback"]:
        key = f"mature_lb{lookback}"
        scores[key], selections[key] = dynamic_score(arrays, int(lookback), selector["horizons"], int(selector["default_before_warmup"]))
        orders[key] = np.argsort(-np.nan_to_num(scores[key], nan=-np.inf), axis=1).astype(np.int32)
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for amount_min in protocol["stage1"]["amount_min"]:
            for mv_min in protocol["stage1"]["total_mv_min"]:
                for top_n in protocol["stage1"]["top_n"]:
                    for entry_rank_min in protocol["stage1"]["entry_rank_min"]:
                        profile = core.Profile(blend_id, amount_min, mv_min, top_n, entry_rank_min, **fixed)
                        daily = engine.simulate(arrays, scores[blend_id], orders[blend_id], profile, protocol["observation_end"])
                        rows.append({"case_id": profile.profile_id, **asdict(profile), **evaluate(daily, protocol["year_folds"])})
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
                            profile = core.Profile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), min_hold, max_hold, sell_rank_below, replacement_advantage, invested_ratio)
                            daily = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, protocol["observation_end"])
                            rows.append({"case_id": profile.profile_id, "seed_case_id": str(seed.case_id), **asdict(profile), **evaluate(daily, protocol["year_folds"])})
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
        profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
        _, actions = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, protocol["observation_end"], record_actions=True)
        key = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        path = ACTION_DIR / f"{profile.profile_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": profile.profile_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    selection_counts = {key: {str(h): int((values == h).sum()) for h in selector["horizons"]} for key, values in selections.items()}
    best = frozen.head(1).to_dict("records")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "horizon_selection_counts": selection_counts, "best": best, "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    best_line = "无逐年为正候选。" if not best else f"最佳候选 `{best[0]['case_id']}`：本地观察期累计收益 {best[0]['full_cumulative_return']:.2%}，Sharpe {best[0]['full_sharpe']:.3f}，最大回撤 {best[0]['full_max_drawdown']:.2%}。"
    REPORT_PATH.write_text("\n".join(["# 成熟期限自适应研究结论", "", "- 参数选择区间：`20220606-20251231`。", "- 期限选择只使用已经成熟的历史收益观测。", "- `2026` 未用于参数选择。", "- 本地仅作预筛，正式收益以掘金为准。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", best_line, "", "本轮为 research-only，未修改生产策略或正式信号。"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
