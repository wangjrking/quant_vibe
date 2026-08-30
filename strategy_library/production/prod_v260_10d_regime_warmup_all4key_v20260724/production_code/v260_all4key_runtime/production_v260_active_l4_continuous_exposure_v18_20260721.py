# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_best_corrected_v6_20260721 as corrected
from . import production_v260_active_l4_lagged_health_gate_v15_20260721 as health
from . import production_v260_active_l4_yearly_robust_v11_20260721 as yearly
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_continuous_exposure_v18_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "连续仓位缩放研究结论.md"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


@dataclass(frozen=True)
class ExposureProfile:
    blend_id: str
    amount_min: int
    mv_min: int
    top_n: int
    entry_rank_min: float
    health_lookback: int
    floor_fraction: float
    scale_denominator: float
    min_hold: int
    max_hold: int
    sell_rank_below: float
    replacement_advantage: float
    invested_ratio: float

    @property
    def case_id(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "ce_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exposure_series(quality, lookback, floor_fraction, denominator):
    result = np.zeros(len(quality), dtype=np.float32)
    minimum = max(5, lookback // 4)
    for t in range(len(quality)):
        end = t - 1
        start = max(0, end - lookback)
        values = quality[start:end]
        values = values[np.isfinite(values)]
        if len(values) >= minimum:
            mean_quality = float(np.mean(values))
            result[t] = float(np.clip(floor_fraction + mean_quality / denominator, floor_fraction, 1.0))
    return result


def simulate(arrays, score, order, exposure, profile, end_signal_date, record_actions=False):
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = 700_000.0
    shares, entry_index, last_price = {}, {}, {}
    previous_equity = cash
    rows, actions = [], []
    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_signal_date:
            break
        buy_date = str(dates[t + 1])
        opens = arrays["buy_open"][t]
        pre_close = arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        universe = arrays["signal_clean"][t] & arrays["buy_clean"][t] & valid_open & np.isfinite(score[t]) & (score[t] >= profile.entry_rank_min) & (arrays["amount"][t] >= profile.amount_min) & (arrays["total_mv"][t] >= profile.mv_min) & (arrays["listed_days"][t] >= 60)
        universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]
        best_unheld_score = max((float(score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        target_positions = int(math.floor(profile.top_n * float(exposure[t]) + 1e-9))
        target_positions = max(min(target_positions, profile.top_n), 0)
        normal_sells = []
        for idx in list(shares):
            age = t - entry_index[idx]
            current_score = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            rank_exit = age >= profile.min_hold and current_score < profile.sell_rank_below and best_unheld_score - current_score >= profile.replacement_advantage
            if age >= profile.max_hold or rank_exit:
                normal_sells.append(idx)
        remaining = [idx for idx in shares if idx not in normal_sells]
        excess = max(len(remaining) - target_positions, 0)
        scale_sells = sorted((idx for idx in remaining if t - entry_index[idx] >= profile.min_hold), key=lambda idx: (float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf, stocks[idx]))[:excess]
        turnover, trades, sell_failed = 0.0, 0, False
        for idx in normal_sells + scale_sells:
            if not valid_open[idx]:
                sell_failed = True
                continue
            rate = corrected.corrected_limit_rate(arrays, t, idx, board_rate)
            if opens[idx] <= pre_close[idx] * (1.0 - rate) * 1.005:
                sell_failed = True
                continue
            gross = float(shares[idx] * opens[idx])
            slip = core.adaptive_slippage(gross, arrays["amount"][t, idx], "sell")
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx]), "exposure_fraction": float(exposure[t])})
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)
        slots = 0 if sell_failed else max(target_positions - len(shares), 0)
        target_value = equity_before * profile.invested_ratio / profile.top_n
        for idx in ranked:
            if slots <= 0 or idx in shares:
                continue
            gross_budget = min(target_value, cash / 1.001)
            if gross_budget < 1000.0:
                break
            slip = core.adaptive_slippage(gross_budget, arrays["amount"][t, idx], "buy")
            buy_price = float(opens[idx]) * (1.0 + slip)
            quantity = gross_budget / (buy_price * (1.0 + 0.0003))
            spend = quantity * buy_price * (1.0 + 0.0003)
            if quantity <= 0 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx] = quantity
            entry_index[idx] = t
            last_price[idx] = float(opens[idx])
            turnover += quantity * float(opens[idx])
            trades += 1
            slots -= 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": profile.invested_ratio / profile.top_n, "execution_open_raw": float(opens[idx]), "exposure_fraction": float(exposure[t])})
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades, "target_exposure": float(exposure[t])})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions)) if record_actions else daily


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
    qualities = {key: health.shadow_quality(arrays, score) for key, score in scores.items()}
    exposures = {(key, lookback, floor, denom): exposure_series(qualities[key], lookback, floor, denom) for key in scores for lookback in protocol["health_to_exposure"]["rolling_lookback"] for floor in protocol["health_to_exposure"]["floor_fraction"] for denom in protocol["health_to_exposure"]["scale_denominator"]}
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for lookback in protocol["health_to_exposure"]["rolling_lookback"]:
            for floor in protocol["health_to_exposure"]["floor_fraction"]:
                for denom in protocol["health_to_exposure"]["scale_denominator"]:
                    for amount_min in protocol["stage1"]["amount_min"]:
                        for top_n in protocol["stage1"]["top_n"]:
                            for entry_rank_min in protocol["stage1"]["entry_rank_min"]:
                                profile = ExposureProfile(blend_id, amount_min, 200000, top_n, entry_rank_min, lookback, floor, denom, **fixed)
                                daily = simulate(arrays, scores[blend_id], orders[blend_id], exposures[(blend_id, lookback, floor, denom)], profile, protocol["observation_end"])
                                rows.append({"case_id": profile.case_id, **asdict(profile), **evaluate(daily, protocol["year_folds"])})
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
                if max_hold <= min_hold:
                    continue
                for sell_rank_below in grid["sell_rank_below"]:
                    for replacement_advantage in grid["replacement_advantage"]:
                        for invested_ratio in grid["invested_ratio"]:
                            profile = ExposureProfile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), int(seed.health_lookback), float(seed.floor_fraction), float(seed.scale_denominator), min_hold, max_hold, sell_rank_below, replacement_advantage, invested_ratio)
                            daily = simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], exposures[(profile.blend_id, profile.health_lookback, profile.floor_fraction, profile.scale_denominator)], profile, protocol["observation_end"])
                            rows.append({"case_id": profile.case_id, "seed_case_id": str(seed.case_id), **asdict(profile), **evaluate(daily, protocol["year_folds"])})
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
        profile = ExposureProfile(**{key: item[key] for key in ExposureProfile.__dataclass_fields__})
        _, actions = simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], exposures[(profile.blend_id, profile.health_lookback, profile.floor_fraction, profile.scale_denominator)], profile, protocol["observation_end"], record_actions=True)
        action_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_hash in seen:
            continue
        seen.add(action_hash)
        path = ACTION_DIR / f"{profile.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": profile.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    best = frozen.head(1).to_dict("records")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "best": best, "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    best_line = "无逐年为正候选。" if not best else f"最佳候选 `{best[0]['case_id']}`：本地观察期累计收益 {best[0]['full_cumulative_return']:.2%}，Sharpe {best[0]['full_sharpe']:.3f}，最大回撤 {best[0]['full_max_drawdown']:.2%}。"
    REPORT_PATH.write_text("\n".join(["# 连续仓位缩放研究结论", "", "- 参数选择仅使用 `20220606-20251231`。", "- 仓位比例只由严格滞后的成熟历史健康度决定。", "- `2026` 未参与筛选。", "- 本地只作预筛，正式结果以掘金为准。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", best_line, "", "本轮为 research-only，未修改生产策略或正式信号。"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
