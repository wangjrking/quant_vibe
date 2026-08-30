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
import research_active_l4_agreement_health_v16_20260721 as candidate_health
import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_market_state_v26_20260721 as market
import research_active_l4_yearly_robust_v11_20260721 as yearly
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_independent_exit_v28_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "独立卖出评分研究结论.md"


@dataclass(frozen=True)
class ExitProfile:
    exit_score_id: str
    amount_min: int
    mv_min: int
    top_n: int
    entry_rank_min: float
    sell_score_below: float
    min_hold: int
    max_hold: int
    replacement_advantage: float
    invested_ratio: float


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_id(profile):
    raw = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"))
    return "ix_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_exit_score(arrays, buy_score, rule):
    kind = rule["type"]
    if kind == "buy_blend":
        return buy_score.astype(np.float32)
    if kind == "rank_10d":
        return arrays["rank_10d"].astype(np.float32)
    if kind == "geometric":
        r10 = np.clip(arrays["rank_10d"], 1e-6, 1.0)
        r5 = np.clip(arrays["rank_5d"], 1e-6, 1.0)
        return np.exp(rule["w10"] * np.log(r10) + rule["w5"] * np.log(r5)).astype(np.float32)
    if kind == "minimum":
        values = [arrays[f"rank_{h}d"] for h in rule["horizons"]]
        result = values[0].copy()
        for value in values[1:]:
            result = np.minimum(result, value)
        return result.astype(np.float32)
    raise ValueError(kind)


def evaluate(daily, folds):
    result = yearly.evaluate(daily, folds)
    result["min_year_cumulative_return"] = min(float(result[f"{fold['id']}_cumulative_return"]) for fold in folds)
    return result


def dual_track(frame, return_count, sharpe_count):
    eligible = frame[frame.eligible & frame.all_year_positive]
    by_return = eligible.sort_values(["full_cumulative_return", "full_sharpe", "min_year_cumulative_return", "case_id"], ascending=[False, False, False, True]).head(return_count).assign(promotion_track="return")
    by_sharpe = eligible.sort_values(["full_sharpe", "full_cumulative_return", "min_year_cumulative_return", "case_id"], ascending=[False, False, False, True]).head(sharpe_count).assign(promotion_track="sharpe")
    return pd.concat([by_return, by_sharpe], ignore_index=True).drop_duplicates("case_id")


def simulate(arrays, masked, buy_score, buy_order, exit_score, exposure, profile, end_signal_date, record_actions=False):
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
        universe = masked["signal_clean"][t] & masked["buy_clean"][t] & valid_open & np.isfinite(buy_score[t]) & (buy_score[t] >= profile.entry_rank_min) & (arrays["amount"][t] >= profile.amount_min) & (arrays["total_mv"][t] >= profile.mv_min) & (arrays["listed_days"][t] >= 60)
        universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in buy_order[t] if universe[int(idx)]]
        best_unheld_exit = max((float(exit_score[t, idx]) for idx in ranked if idx not in shares and np.isfinite(exit_score[t, idx])), default=-np.inf)
        target_positions = int(math.floor(profile.top_n * float(exposure[t]) + 1e-9))
        target_positions = max(min(target_positions, profile.top_n), 0)
        normal_sells = []
        for idx in list(shares):
            age = t - entry_index[idx]
            current_exit = float(exit_score[t, idx]) if np.isfinite(exit_score[t, idx]) else -np.inf
            score_exit = age >= profile.min_hold and current_exit < profile.sell_score_below and best_unheld_exit - current_exit >= profile.replacement_advantage
            if age >= profile.max_hold or score_exit:
                normal_sells.append(idx)
        remaining = [idx for idx in shares if idx not in normal_sells]
        excess = max(len(remaining) - target_positions, 0)
        scale_sells = sorted((idx for idx in remaining if t - entry_index[idx] >= profile.min_hold), key=lambda idx: (float(exit_score[t, idx]) if np.isfinite(exit_score[t, idx]) else -np.inf, stocks[idx]))[:excess]
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
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx]), "exit_score_id": profile.exit_score_id})
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
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": profile.invested_ratio / profile.top_n, "execution_open_raw": float(opens[idx]), "exit_score_id": profile.exit_score_id})
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions)) if record_actions else daily


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    buy_score = core.blend_scores(arrays, protocol["buy_score"])
    buy_order = np.argsort(-np.nan_to_num(buy_score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement.masked_arrays(arrays, protocol["agreement"])
    exit_scores = {rule["id"]: build_exit_score(arrays, buy_score, rule) for rule in protocol["exit_scores"]}
    health_rule = protocol["model_health"]
    quality = candidate_health.candidate_quality(masked, buy_score, buy_order, int(health_rule["shadow_top_n"]))
    model_exposure = continuous.exposure_series(quality, int(health_rule["rolling_lookback"]), float(health_rule["floor_fraction"]), float(health_rule["scale_denominator"]))
    market_rule = protocol["market_state"]
    market_return = market.market_open_return(arrays)
    market_exp = market.market_exposure(market_return, int(market_rule["lookback"]), float(market_rule["mean_return_threshold"]), float(market_rule["risk_off_floor"]))
    exposure = np.minimum(market_exp, model_exposure).astype(np.float32)

    rows = []
    grid = protocol["stage1"]
    fixed = grid["fixed"]
    for exit_score_id, exit_score in exit_scores.items():
        for amount_min in grid["amount_min"]:
            for mv_min in grid["total_mv_min"]:
                for entry_rank_min in grid["entry_rank_min"]:
                    for sell_score_below in grid["sell_score_below"]:
                        profile = ExitProfile(exit_score_id, amount_min, mv_min, int(grid["top_n"]), entry_rank_min, sell_score_below, int(fixed["min_hold"]), int(fixed["max_hold"]), float(fixed["replacement_advantage"]), 1.0)
                        daily = simulate(arrays, masked, buy_score, buy_order, exit_score, exposure, profile, protocol["observation_end"])
                        rows.append({"case_id": case_id(profile), **asdict(profile), **evaluate(daily, protocol["year_folds"])})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    seeds = dual_track(stage1, int(grid["promote_return"]), int(grid["promote_sharpe"]))

    rows = []
    for _, seed in seeds.iterrows():
        for min_hold in protocol["stage2"]["min_hold"]:
            for max_hold in protocol["stage2"]["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for advantage in protocol["stage2"]["replacement_advantage"]:
                    profile = ExitProfile(str(seed.exit_score_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), float(seed.sell_score_below), int(min_hold), int(max_hold), float(advantage), 1.0)
                    daily = simulate(arrays, masked, buy_score, buy_order, exit_scores[profile.exit_score_id], exposure, profile, protocol["observation_end"])
                    rows.append({"case_id": case_id(profile), "seed_case_id": str(seed.case_id), **asdict(profile), **evaluate(daily, protocol["year_folds"])})
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
    stage2 = stage2.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = dual_track(stage2, int(protocol["stage2"]["promote_return"]), int(protocol["stage2"]["promote_sharpe"]))
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for item in payload["profiles"]:
        profile = ExitProfile(**{key: item[key] for key in ExitProfile.__dataclass_fields__})
        _, actions = simulate(arrays, masked, buy_score, buy_order, exit_scores[profile.exit_score_id], exposure, profile, protocol["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = ACTION_DIR / f"{item['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item["case_id"], "promotion_track": item["promotion_track"], "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "content_sha256": content_hash, "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "best_return": frozen.sort_values("full_cumulative_return", ascending=False).head(1).to_dict("records"), "best_sharpe": frozen.sort_values("full_sharpe", ascending=False).head(1).to_dict("records"), "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(["# 独立卖出评分研究结论", "", "- 参数选择仅使用 `20220606-20251231`。", "- 卖出基于持仓与当日新候选的同口径评分比较，不把未入选直接等同于卖出。", "- 2026数据未参与筛选。", "- 本地只作预筛，正式结果以掘金为准。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结候选 {len(frozen)} 组，独立交易路径 {len(action_rows)} 条。", "", "本轮为research-only，未修改生产策略或正式信号。"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"best_return", "best_sharpe"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
