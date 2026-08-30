# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_agreement_health_v16_20260721 as health
import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_buy_coverage_v39_20260721 as v39
import research_active_l4_candidate_strength_v30_20260721 as v30
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_independent_sell_v36_20260721 as v36
import research_active_l4_market_state_v26_20260721 as market
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_exit_cadence_v47_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


@dataclass(frozen=True)
class CadenceExit:
    review_interval: int
    min_hold: int
    max_hold: int
    sell_rank_below: float
    replacement_advantage: float

    @property
    def case_id(self):
        return f"r{self.review_interval}_h{self.min_hold}_{self.max_hold}_s{int(self.sell_rank_below*100):02d}_a{int(self.replacement_advantage*100):02d}"


def ratio_weights(count, scheme):
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [0.90, 0.10]


def simulate(arrays, masked, buy_score, order, exposure, buy_profile, exit_rule, end_date, record_actions=False):
    dates, stocks = arrays["dates"].astype(str), arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    sell_score = arrays["rank_5d"]
    cash, previous_equity = 700_000.0, 700_000.0
    shares, entry_index, last_price = {}, {}, {}
    rows, actions = [], []
    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_date:
            break
        buy_date = str(dates[t + 1])
        opens, pre_close = arrays["buy_open"][t], arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        universe = masked["signal_clean"][t] & masked["buy_clean"][t] & valid_open & np.isfinite(buy_score[t])
        universe &= (buy_score[t] >= buy_profile.entry_rank_min) & (arrays["amount"][t] >= buy_profile.amount_min) & (arrays["total_mv"][t] >= buy_profile.mv_min) & (arrays["listed_days"][t] >= 60)
        universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]
        target_positions = max(min(int(math.floor(buy_profile.top_n * float(exposure[t]) + 1e-9)), buy_profile.top_n), 0)
        best_unheld = max((float(buy_score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        normal_sells = []
        for idx in list(shares):
            age = t - entry_index[idx]
            held_exit = float(sell_score[t, idx]) if np.isfinite(sell_score[t, idx]) else -np.inf
            held_buy = float(buy_score[t, idx]) if np.isfinite(buy_score[t, idx]) else -np.inf
            review_due = age >= exit_rule.min_hold and (age - exit_rule.min_hold) % exit_rule.review_interval == 0
            score_exit = review_due and held_exit < exit_rule.sell_rank_below and best_unheld - held_buy >= exit_rule.replacement_advantage
            if age >= exit_rule.max_hold or score_exit:
                normal_sells.append(idx)
        remaining = [idx for idx in shares if idx not in normal_sells]
        excess = max(len(remaining) - target_positions, 0)
        scale_sells = sorted((idx for idx in remaining if t - entry_index[idx] >= exit_rule.min_hold), key=lambda idx: (float(sell_score[t, idx]) if np.isfinite(sell_score[t, idx]) else -np.inf, stocks[idx]))[:excess]
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
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx])})
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)
        slots = 0 if sell_failed else max(target_positions - len(shares), 0)
        if slots > 0:
            candidates = [idx for idx in ranked if idx not in shares][:slots]
            total_target = buy_profile.normal_total_pct * float(exposure[t])
            weights = ratio_weights(len(candidates), buy_profile.weight_scheme)
            for idx, weight in zip(candidates, weights):
                target_pct = total_target * weight
                gross_budget = min(equity_before * target_pct, cash / 1.001)
                if gross_budget < 1000:
                    continue
                slip = core.adaptive_slippage(gross_budget, arrays["amount"][t, idx], "buy")
                buy_price = float(opens[idx]) * (1.0 + slip)
                quantity = gross_budget / (buy_price * (1.0 + 0.0003))
                spend = quantity * buy_price * (1.0 + 0.0003)
                if quantity <= 0 or spend > cash + 1e-6:
                    continue
                cash -= spend
                shares[idx], entry_index[idx], last_price[idx] = quantity, t, float(opens[idx])
                turnover += quantity * float(opens[idx])
                trades += 1
                if record_actions:
                    actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": target_pct, "execution_open_raw": float(opens[idx])})
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions)) if record_actions else daily


def robust_sort(frame):
    return frame.sort_values(["all_year_positive", "min_year_cumulative_return", "median_year_sharpe", "full_cumulative_return", "case_id"], ascending=[False, False, False, False, True])


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
    b = p["fixed_buy"]
    bp = v39.BuyProfile(2, b["entry_rank_min"], b["amount_min"], b["mv_min"], b["normal_total_pct"], False, 0.99, 0.005, 1.0, b["weight_scheme"])
    rows, g = [], p["grid"]
    for review in g["review_interval"]:
        for min_hold in g["min_hold"]:
            for max_hold in g["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for threshold in g["sell_rank_below"]:
                    for advantage in g["replacement_advantage"]:
                        rule = CadenceExit(review, min_hold, max_hold, threshold, advantage)
                        daily = simulate(arrays, masked, buy_score, order, exposure, bp, rule, p["observation_end"])
                        rows.append({"case_id": rule.case_id, **rule.__dict__, **v30.evaluate(daily, p["year_folds"])})
    results = robust_sort(pd.DataFrame(rows))
    results["eligible"] = (results.full_max_drawdown <= 0.40) & (results.full_trades >= 80)
    results.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")
    pool = results[(results.all_year_positive) & results.eligible]
    by_return = pool.sort_values(["full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, True]).head(int(g["promote_return"]))
    by_robust = robust_sort(pool).head(int(g["promote_robust"]))
    frozen = pd.concat([by_return, by_robust], ignore_index=True).drop_duplicates("case_id")
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": v30.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        rule = CadenceExit(int(item.review_interval), int(item.min_hold), int(item.max_hold), float(item.sell_rank_below), float(item.replacement_advantage))
        _, actions = simulate(arrays, masked, buy_score, order, exposure, bp, rule, p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "grid_cases": len(results), "all_year_positive": int(results.all_year_positive.sum()), "eligible_cases": len(pool), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "卖出检查频率研究结论.md").write_text("\n".join(["# 卖出检查频率研究结论", "", "本轮固定买入路径，仅调整独立评分卖出的检查频率和持仓规则。", "", f"固定网格 {len(results)} 组，逐年为正 {int(results.all_year_positive.sum())} 组，强过滤后 {len(pool)} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "2026验证期未打开，生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
