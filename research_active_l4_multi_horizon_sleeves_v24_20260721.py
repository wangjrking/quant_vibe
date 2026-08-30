# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_yearly_robust_v11_20260721 as yearly
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_multi_horizon_sleeves_v24_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "多周期袖套研究结论.md"


@dataclass(frozen=True)
class SleeveProfile:
    allocation_id: str
    amount_min: int
    mv_min: int
    entry_rank_min: float
    support_rank_min: float
    exit_id: str
    replacement_advantage: float

    @property
    def case_id(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "ms_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(daily, folds):
    result = yearly.evaluate(daily, folds)
    result["min_year_cumulative_return"] = min(
        float(result[f"{fold['id']}_cumulative_return"]) for fold in folds
    )
    return result


def dual_track(frame, return_count, sharpe_count):
    eligible = frame[frame.eligible & frame.all_year_positive]
    by_return = eligible.sort_values(
        ["full_cumulative_return", "full_sharpe", "min_year_cumulative_return", "case_id"],
        ascending=[False, False, False, True]
    ).head(return_count).assign(promotion_track="return")
    by_sharpe = eligible.sort_values(
        ["full_sharpe", "full_cumulative_return", "min_year_cumulative_return", "case_id"],
        ascending=[False, False, False, True]
    ).head(sharpe_count).assign(promotion_track="sharpe")
    return pd.concat([by_return, by_sharpe], ignore_index=True).drop_duplicates("case_id")


def simulate(arrays, profile, allocations, exits, sleeve_priority, end_signal_date, record_actions=False):
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    ranks = {h: arrays[f"rank_{h}d"] for h in (3, 5, 10)}
    orders = {
        h: np.argsort(-np.nan_to_num(ranks[h], nan=-np.inf), axis=1).astype(np.int32)
        for h in (3, 5, 10)
    }
    board_rate = core.board_limit_rate(stocks)
    allocation = allocations[profile.allocation_id]
    exit_rule = exits[profile.exit_id]
    weights = {3: float(allocation["w3"]), 5: float(allocation["w5"]), 10: float(allocation["w10"])}

    cash = 700_000.0
    shares, last_price = {}, {}
    sleeve_stock, entry_index = {}, {}
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

        universes, ranked = {}, {}
        for horizon in (3, 5, 10):
            universe = (
                arrays["signal_clean"][t]
                & arrays["buy_clean"][t]
                & valid_open
                & np.isfinite(ranks[horizon][t])
                & (ranks[horizon][t] >= profile.entry_rank_min)
                & (arrays["amount"][t] >= profile.amount_min)
                & (arrays["total_mv"][t] >= profile.mv_min)
                & (arrays["listed_days"][t] >= 60)
            )
            if profile.support_rank_min > 0:
                for other in (3, 5, 10):
                    if other != horizon:
                        universe &= np.isfinite(ranks[other][t]) & (ranks[other][t] >= profile.support_rank_min)
            universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
            universes[horizon] = universe
            ranked[horizon] = [int(idx) for idx in orders[horizon][t] if universe[int(idx)]]

        sell_sleeves = []
        held_indices = set(sleeve_stock.values())
        for horizon, idx in list(sleeve_stock.items()):
            age = t - entry_index[horizon]
            current_score = float(ranks[horizon][t, idx]) if np.isfinite(ranks[horizon][t, idx]) else -np.inf
            best_unheld = max(
                (float(ranks[horizon][t, candidate]) for candidate in ranked[horizon] if candidate not in held_indices),
                default=-np.inf
            )
            min_hold = int(exit_rule["min_hold"][str(horizon)])
            max_hold = int(exit_rule["max_hold"][str(horizon)])
            sell_below = float(exit_rule["sell_rank_below"][str(horizon)])
            rank_exit = (
                age >= min_hold
                and current_score < sell_below
                and best_unheld - current_score >= profile.replacement_advantage
            )
            if age >= max_hold or rank_exit:
                sell_sleeves.append(horizon)

        turnover, trades = 0.0, 0
        for horizon in sell_sleeves:
            idx = sleeve_stock[horizon]
            if not valid_open[idx]:
                continue
            rate = corrected.corrected_limit_rate(arrays, t, idx, board_rate)
            if opens[idx] <= pre_close[idx] * (1.0 - rate) * 1.005:
                continue
            gross = float(shares[idx] * opens[idx])
            slip = core.adaptive_slippage(gross, arrays["amount"][t, idx], "sell")
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append({
                    "signal_date": signal_date, "buy_date": buy_date, "action": "SELL",
                    "stock_code": stocks[idx], "target_pct": 0.0,
                    "execution_open_raw": float(opens[idx]), "sleeve_horizon": horizon
                })
            del shares[idx]
            last_price.pop(idx, None)
            del sleeve_stock[horizon]
            del entry_index[horizon]

        held_indices = set(sleeve_stock.values())
        for horizon in sleeve_priority:
            if weights[horizon] <= 0 or horizon in sleeve_stock:
                continue
            chosen = next((idx for idx in ranked[horizon] if idx not in held_indices), None)
            if chosen is None:
                continue
            gross_budget = min(equity_before * weights[horizon], cash / 1.001)
            if gross_budget < 1000.0:
                continue
            slip = core.adaptive_slippage(gross_budget, arrays["amount"][t, chosen], "buy")
            buy_price = float(opens[chosen]) * (1.0 + slip)
            quantity = gross_budget / (buy_price * (1.0 + 0.0003))
            spend = quantity * buy_price * (1.0 + 0.0003)
            if quantity <= 0 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[chosen] = quantity
            last_price[chosen] = float(opens[chosen])
            sleeve_stock[horizon] = chosen
            entry_index[horizon] = t
            held_indices.add(chosen)
            turnover += quantity * float(opens[chosen])
            trades += 1
            if record_actions:
                actions.append({
                    "signal_date": signal_date, "buy_date": buy_date, "action": "BUY",
                    "stock_code": stocks[chosen], "target_pct": weights[horizon],
                    "execution_open_raw": float(opens[chosen]), "sleeve_horizon": horizon
                })

        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({
            "date": buy_date,
            "return": equity_after / previous_equity - 1.0,
            "equity": equity_after,
            "turnover": turnover / max(equity_before, 1.0),
            "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0,
            "positions": len(shares),
            "trades": trades,
        })
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
    allocations = {item["id"]: item for item in protocol["allocations"]}
    exits = {item["id"]: item for item in protocol["exit_profiles"]}
    priority = [int(value) for value in protocol["sleeve_priority"]]

    rows = []
    stage1_grid = protocol["stage1"]
    for allocation_id in stage1_grid["allocation_ids"]:
        for amount_min in stage1_grid["amount_min"]:
            for mv_min in stage1_grid["total_mv_min"]:
                for entry_rank_min in stage1_grid["entry_rank_min"]:
                    for support_rank_min in stage1_grid["support_rank_min"]:
                        profile = SleeveProfile(
                            allocation_id, amount_min, mv_min, entry_rank_min,
                            support_rank_min, stage1_grid["fixed_exit_id"],
                            stage1_grid["fixed_replacement_advantage"]
                        )
                        daily = simulate(
                            arrays, profile, allocations, exits, priority,
                            protocol["observation_end"]
                        )
                        rows.append({"case_id": profile.case_id, **asdict(profile), **evaluate(daily, protocol["year_folds"])})
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(
        ["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"],
        ascending=[False, False, False, True]
    )
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    seeds = dual_track(stage1, int(stage1_grid["promote_return"]), int(stage1_grid["promote_sharpe"]))

    if seeds.empty:
        pd.DataFrame().to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
        payload = {
            "protocol_sha256": digest(PROTOCOL_PATH),
            "stage1_sha256": digest(STAGE1_PATH),
            "stage2_sha256": digest(STAGE2_PATH),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "profiles": [],
        }
        FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        ACTION_DIR.mkdir(parents=True, exist_ok=True)
        ACTION_MANIFEST.write_text(json.dumps({
            "frozen_sha256": digest(FROZEN_PATH), "actions": []
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "status": "research_only_observation_no_candidate",
            "stage1_cases": len(stage1),
            "stage1_all_year_positive": int(stage1.all_year_positive.sum()),
            "stage2_cases": 0,
            "stage2_all_year_positive": 0,
            "frozen_candidates": 0,
            "unique_juejin_paths": 0,
            "best_stage1_by_return": stage1.sort_values(
                "full_cumulative_return", ascending=False
            ).head(1).to_dict("records"),
            "known_2026_used_for_selection": False,
            "production_changed": False,
        }
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_PATH.write_text("\n".join([
            "# 多周期袖套研究结论", "",
            "- 参数选择仅使用 `20220606-20251231`。",
            "- 第一阶段没有候选同时满足逐年为正、最大回撤不超过40%和交易数门槛。",
            "- 因此未进入第二阶段，也未提交掘金回测。",
            "- 2026数据未参与筛选，生产策略未修改。"
        ]), encoding="utf-8")
        print(json.dumps({
            key: value for key, value in summary.items()
            if key != "best_stage1_by_return"
        }, ensure_ascii=False))
        return

    rows = []
    stage2_grid = protocol["stage2"]
    for _, seed in seeds.iterrows():
        for exit_id in stage2_grid["exit_ids"]:
            for replacement_advantage in stage2_grid["replacement_advantage"]:
                profile = SleeveProfile(
                    str(seed.allocation_id), int(seed.amount_min), int(seed.mv_min),
                    float(seed.entry_rank_min), float(seed.support_rank_min), exit_id,
                    replacement_advantage
                )
                daily = simulate(
                    arrays, profile, allocations, exits, priority,
                    protocol["observation_end"]
                )
                rows.append({
                    "case_id": profile.case_id, "seed_case_id": str(seed.case_id),
                    **asdict(profile), **evaluate(daily, protocol["year_folds"])
                })
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
    stage2 = stage2.sort_values(
        ["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"],
        ascending=[False, False, False, True]
    )
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = dual_track(stage2, int(stage2_grid["promote_return"]), int(stage2_grid["promote_sharpe"]))

    payload = {
        "protocol_sha256": digest(PROTOCOL_PATH),
        "stage1_sha256": digest(STAGE1_PATH),
        "stage2_sha256": digest(STAGE2_PATH),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profiles": frozen.to_dict("records"),
    }
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for item in payload["profiles"]:
        profile = SleeveProfile(**{key: item[key] for key in SleeveProfile.__dataclass_fields__})
        _, actions = simulate(
            arrays, profile, allocations, exits, priority,
            protocol["observation_end"], record_actions=True
        )
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = ACTION_DIR / f"{profile.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({
            "case_id": profile.case_id,
            "promotion_track": item["promotion_track"],
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(path),
            "content_sha256": content_hash,
            "rows": len(actions),
            "buy_rows": int((actions.action == "BUY").sum()),
            "sell_rows": int((actions.action == "SELL").sum()),
        })
    ACTION_MANIFEST.write_text(json.dumps({
        "frozen_sha256": digest(FROZEN_PATH), "actions": action_rows
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "research_only_observation",
        "stage1_cases": len(stage1),
        "stage1_all_year_positive": int(stage1.all_year_positive.sum()),
        "stage2_cases": len(stage2),
        "stage2_all_year_positive": int(stage2.all_year_positive.sum()),
        "frozen_candidates": len(frozen),
        "unique_juejin_paths": len(action_rows),
        "best_return": frozen.sort_values("full_cumulative_return", ascending=False).head(1).to_dict("records"),
        "best_sharpe": frozen.sort_values("full_sharpe", ascending=False).head(1).to_dict("records"),
        "known_2026_used_for_selection": False,
        "production_changed": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join([
        "# 多周期袖套研究结论", "",
        "- 参数选择仅使用 `20220606-20251231`。",
        "- 3D、5D、10D袖套持仓互斥，各自按正式评分退出。",
        "- 2026数据未参与筛选。",
        "- 本地只作预筛，正式结果以掘金为准。", "",
        f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。",
        f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。",
        f"冻结候选 {len(frozen)} 组，独立交易路径 {len(action_rows)} 条。", "",
        "本轮为research-only，未修改生产策略或正式信号。"
    ]), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in summary.items()
        if key not in {"best_return", "best_sharpe"}
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
