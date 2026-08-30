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
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_candidate_strength_v30_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "候选强度分层研究结论.md"


@dataclass(frozen=True)
class StrengthProfile:
    amount_min: int
    mv_min: int
    entry_rank_min: float
    strong_top_score_min: float
    strong_gap_min: float
    strong_target_pct: float
    normal_total_pct: float
    min_hold: int
    max_hold: int
    sell_rank_below: float
    replacement_advantage: float


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_id(profile):
    raw = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"))
    return "cs_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluate(daily, folds):
    result = yearly.evaluate(daily, folds)
    result["min_year_cumulative_return"] = min(float(result[f"{fold['id']}_cumulative_return"]) for fold in folds)
    return result


def dual_track(frame, return_count, sharpe_count):
    eligible = frame[frame.eligible & frame.all_year_positive]
    by_return = eligible.sort_values(["full_cumulative_return", "full_sharpe", "min_year_cumulative_return", "case_id"], ascending=[False, False, False, True]).head(return_count).assign(promotion_track="return")
    by_sharpe = eligible.sort_values(["full_sharpe", "full_cumulative_return", "min_year_cumulative_return", "case_id"], ascending=[False, False, False, True]).head(sharpe_count).assign(promotion_track="sharpe")
    return pd.concat([by_return, by_sharpe], ignore_index=True).drop_duplicates("case_id")


def simulate(arrays, masked, score, order, exposure, profile, end_signal_date, record_actions=False):
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = 700_000.0
    shares, entry_index, last_price = {}, {}, {}
    previous_equity = cash
    rows, actions = [], []
    strong_days = 0
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
        universe = masked["signal_clean"][t] & masked["buy_clean"][t] & valid_open & np.isfinite(score[t]) & (score[t] >= profile.entry_rank_min) & (arrays["amount"][t] >= profile.amount_min) & (arrays["total_mv"][t] >= profile.mv_min) & (arrays["listed_days"][t] >= 60)
        universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]
        strong = len(ranked) >= 2 and float(score[t, ranked[0]]) >= profile.strong_top_score_min and float(score[t, ranked[0]] - score[t, ranked[1]]) >= profile.strong_gap_min
        strong_days += int(strong)
        target_positions = 1 if strong and exposure[t] > 0 else int(math.floor(2 * float(exposure[t]) + 1e-9))
        target_positions = max(min(target_positions, 2), 0)
        best_unheld = max((float(score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        normal_sells = []
        for idx in list(shares):
            age = t - entry_index[idx]
            current = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            score_exit = age >= profile.min_hold and current < profile.sell_rank_below and best_unheld - current >= profile.replacement_advantage
            if age >= profile.max_hold or score_exit:
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
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx]), "strength_mode": "strong" if strong else "normal"})
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)
        slots = 0 if sell_failed else max(target_positions - len(shares), 0)
        target_pct = profile.strong_target_pct * float(exposure[t]) if strong else profile.normal_total_pct / 2.0
        for idx in ranked:
            if slots <= 0 or idx in shares:
                continue
            gross_budget = min(equity_before * target_pct, cash / 1.001)
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
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": target_pct, "execution_open_raw": float(opens[idx]), "strength_mode": "strong" if strong else "normal"})
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades, "strong_mode": int(strong)})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    if record_actions:
        return daily, pd.DataFrame(actions)
    return daily


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score = core.blend_scores(arrays, protocol["score"])
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement.masked_arrays(arrays, protocol["agreement"])
    health_rule = protocol["model_health"]
    quality = candidate_health.candidate_quality(masked, score, order, int(health_rule["shadow_top_n"]))
    model_exposure = continuous.exposure_series(quality, int(health_rule["rolling_lookback"]), float(health_rule["floor_fraction"]), float(health_rule["scale_denominator"]))
    market_rule = protocol["market_state"]
    market_exp = market.market_exposure(market.market_open_return(arrays), int(market_rule["lookback"]), float(market_rule["mean_return_threshold"]), float(market_rule["risk_off_floor"]))
    exposure = np.minimum(market_exp, model_exposure).astype(np.float32)
    grid = protocol["stage1"]
    fixed = grid["fixed"]
    rows = []
    for amount_min in grid["amount_min"]:
        for mv_min in grid["total_mv_min"]:
            for entry_rank_min in grid["entry_rank_min"]:
                for top_score in grid["strong_top_score_min"]:
                    for gap in grid["strong_gap_min"]:
                        for strong_pct in grid["strong_target_pct"]:
                            for normal_pct in grid["normal_total_pct"]:
                                profile = StrengthProfile(amount_min, mv_min, entry_rank_min, top_score, gap, strong_pct, normal_pct, int(fixed["min_hold"]), int(fixed["max_hold"]), float(fixed["sell_rank_below"]), float(fixed["replacement_advantage"]))
                                daily = simulate(arrays, masked, score, order, exposure, profile, protocol["observation_end"])
                                metrics = evaluate(daily, protocol["year_folds"])
                                metrics["strong_days"] = int(daily["strong_mode"].sum())
                                rows.append({"case_id": case_id(profile), **asdict(profile), **metrics})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    seeds = dual_track(stage1, int(grid["promote_return"]), int(grid["promote_sharpe"]))
    rows = []
    for _, seed in seeds.iterrows():
        for exit_rule in protocol["stage2_exit_profiles"]:
            profile = StrengthProfile(int(seed.amount_min), int(seed.mv_min), float(seed.entry_rank_min), float(seed.strong_top_score_min), float(seed.strong_gap_min), float(seed.strong_target_pct), float(seed.normal_total_pct), int(exit_rule["min_hold"]), int(exit_rule["max_hold"]), float(exit_rule["sell_rank_below"]), float(exit_rule["replacement_advantage"]))
            daily = simulate(arrays, masked, score, order, exposure, profile, protocol["observation_end"])
            metrics = evaluate(daily, protocol["year_folds"])
            metrics["strong_days"] = int(daily["strong_mode"].sum())
            rows.append({"case_id": case_id(profile), "seed_case_id": str(seed.case_id), "exit_profile_id": exit_rule["id"], **asdict(profile), **metrics})
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
        profile = StrengthProfile(**{key: item[key] for key in StrengthProfile.__dataclass_fields__})
        _, actions = simulate(arrays, masked, score, order, exposure, profile, protocol["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = ACTION_DIR / f"{item['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item["case_id"], "promotion_track": item["promotion_track"], "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "content_sha256": content_hash, "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum()), "max_positions": 2})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()), "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "best_return": frozen.sort_values("full_cumulative_return", ascending=False).head(1).to_dict("records"), "best_sharpe": frozen.sort_values("full_sharpe", ascending=False).head(1).to_dict("records"), "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(["# 候选强度分层研究结论", "", "- 参数选择仅使用 `20220606-20251231`。", "- 强弱只由T日正式评分绝对值与前两名差距决定。", "- 2026数据未参与筛选。", "- 本地只作预筛，正式结果以掘金为准。", "", f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。", f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。", f"冻结候选 {len(frozen)} 组，独立交易路径 {len(action_rows)} 条。", "", "本轮为research-only，未修改生产策略或正式信号。"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"best_return", "best_sharpe"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
