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

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_agreement_health_v16_20260721 as health
import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_candidate_strength_v30_20260721 as v30
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_independent_sell_v36_20260721 as v36
import research_active_l4_market_state_v26_20260721 as market
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_buy_coverage_v39_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


@dataclass(frozen=True)
class BuyProfile:
    top_n: int
    entry_rank_min: float
    amount_min: int
    mv_min: int
    normal_total_pct: float
    strong_single_enabled: bool
    strong_top_score_min: float
    strong_gap_min: float
    strong_target_pct: float
    weight_scheme: str

    @property
    def case_id(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "bc_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def slot_weights(count, scheme):
    if count <= 0:
        return []
    if scheme == "equal" or count == 1:
        return [1.0 / count] * count
    raw = np.arange(count, 0, -1, dtype=float)
    raw /= raw.sum()
    return raw.tolist()


def simulate(arrays, masked, buy_score, order, exposure, buy_profile, exit_profile, end_date, record_actions=False):
    dates, stocks = arrays["dates"].astype(str), arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    sell_score = v36.exit_score_array(arrays, buy_score, exit_profile.exit_score_source)
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
        strong = buy_profile.strong_single_enabled and len(ranked) >= 2 and float(buy_score[t, ranked[0]]) >= buy_profile.strong_top_score_min and float(buy_score[t, ranked[0]] - buy_score[t, ranked[1]]) >= buy_profile.strong_gap_min
        target_positions = 1 if strong and exposure[t] > 0 else int(math.floor(buy_profile.top_n * float(exposure[t]) + 1e-9))
        target_positions = max(min(target_positions, buy_profile.top_n), 0)
        best_unheld = max((float(buy_score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        normal_sells = []
        for idx in list(shares):
            age = t - entry_index[idx]
            held_exit = float(sell_score[t, idx]) if np.isfinite(sell_score[t, idx]) else -np.inf
            held_buy = float(buy_score[t, idx]) if np.isfinite(buy_score[t, idx]) else -np.inf
            score_exit = age >= exit_profile.min_hold and held_exit < exit_profile.sell_rank_below and best_unheld - held_buy >= exit_profile.replacement_advantage
            if age >= exit_profile.max_hold or score_exit:
                normal_sells.append(idx)
        remaining = [idx for idx in shares if idx not in normal_sells]
        excess = max(len(remaining) - target_positions, 0)
        scale_sells = sorted((idx for idx in remaining if t - entry_index[idx] >= exit_profile.min_hold), key=lambda idx: (float(sell_score[t, idx]) if np.isfinite(sell_score[t, idx]) else -np.inf, stocks[idx]))[:excess]
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
            total_target = buy_profile.strong_target_pct * float(exposure[t]) if strong else buy_profile.normal_total_pct * float(exposure[t])
            weights = slot_weights(len(candidates), buy_profile.weight_scheme)
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
    exit_profile = v36.ExitProfile(**p["fixed_exit"])
    g, rows = p["grid"], []
    for top_n in g["top_n"]:
        for entry in g["entry_rank_min"]:
            for amount in g["amount_min"]:
                for mv in g["mv_min"]:
                    for total in g["normal_total_pct"]:
                        for strong in g["strong_single_enabled"]:
                            profile = BuyProfile(top_n, entry, amount, mv, total, strong, g["fixed"]["strong_top_score_min"], g["fixed"]["strong_gap_min"], g["fixed"]["strong_target_pct"], g["fixed"]["weight_scheme"])
                            daily = simulate(arrays, masked, buy_score, order, exposure, profile, exit_profile, p["observation_end"])
                            metrics = v30.evaluate(daily, p["year_folds"])
                            rows.append({"case_id": profile.case_id, **asdict(profile), **metrics})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage1.to_csv(OUT / "stage1.csv", index=False, encoding="utf-8-sig")
    seeds = v30.dual_track(stage1, int(g["promote_return"]), int(g["promote_sharpe"]))
    rows = []
    for _, seed in seeds.iterrows():
        for scheme in p["stage2_weight_schemes"]:
            profile = BuyProfile(int(seed.top_n), float(seed.entry_rank_min), int(seed.amount_min), int(seed.mv_min), float(seed.normal_total_pct), bool(seed.strong_single_enabled), float(seed.strong_top_score_min), float(seed.strong_gap_min), float(seed.strong_target_pct), scheme)
            daily = simulate(arrays, masked, buy_score, order, exposure, profile, exit_profile, p["observation_end"])
            metrics = v30.evaluate(daily, p["year_folds"])
            rows.append({"case_id": profile.case_id, **asdict(profile), **metrics})
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
        profile = BuyProfile(**{key: item[key] for key in BuyProfile.__dataclass_fields__})
        _, actions = simulate(arrays, masked, buy_score, order, exposure, profile, exit_profile, p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "top_n": int(item.top_n), "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": v30.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": int(item.top_n)})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "买入覆盖与持仓结构研究结论.md").write_text("\n".join(["# 买入覆盖与持仓结构研究结论", "", "本轮只调整买入覆盖、持仓数量与仓位，卖出规则固定为V36。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结 {len(action_rows)} 条独立掘金路径。", "", "2026验证期未打开，生产资产未修改。"]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
