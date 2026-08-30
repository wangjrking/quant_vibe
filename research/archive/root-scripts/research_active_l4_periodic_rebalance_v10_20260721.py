from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_best_corrected_v6_20260721 as execution
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_periodic_rebalance_v10_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
GRID_PATH = REPORT_DIR / "observation_grid.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Profile:
    blend_id: str
    amount_min: int
    mv_min: int
    top_n: int
    rebalance_every: int
    entry_rank_min: float
    exit_rank_below: float
    max_hold: int
    invested_ratio: float

    @property
    def case_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "pr_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def simulate(arrays, score, order, profile: Profile, end_signal_date: str, record_actions: bool = False):
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = 700_000.0
    shares: dict[int, float] = {}
    entry_index: dict[int, int] = {}
    last_price: dict[int, float] = {}
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
        turnover = 0.0
        trades = 0
        if t % profile.rebalance_every == 0:
            universe = (
                arrays["signal_clean"][t]
                & arrays["buy_clean"][t]
                & valid_open
                & np.isfinite(score[t])
                & (score[t] >= profile.entry_rank_min)
                & (arrays["amount"][t] >= profile.amount_min)
                & (arrays["total_mv"][t] >= profile.mv_min)
                & (arrays["listed_days"][t] >= 60)
            )
            limit_up = opens >= pre_close * (1.0 + board_rate) * 0.995
            universe &= ~limit_up
            ranked = [int(idx) for idx in order[t] if universe[int(idx)]]
            sell_list = []
            for idx in list(shares):
                age = t - entry_index[idx]
                current_score = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
                if current_score < profile.exit_rank_below or age >= profile.max_hold:
                    sell_list.append(idx)
            sell_failed = False
            for idx in sell_list:
                if not valid_open[idx]:
                    sell_failed = True
                    continue
                rate = execution.corrected_limit_rate(arrays, t, idx, board_rate)
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
            if not sell_failed:
                slots = max(profile.top_n - len(shares), 0)
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
                        actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": profile.invested_ratio / profile.top_n, "execution_open_raw": float(opens[idx])})
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions)) if record_actions else daily


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("frozen protocol or input drift")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {item["id"]: core.blend_scores(arrays, item) for item in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    grid = protocol["parameter_grid"]
    rows = []
    for values in itertools.product(
        scores,
        grid["amount_min"],
        grid["total_mv_min"],
        grid["top_n"],
        grid["rebalance_every"],
        grid["entry_rank_min"],
        grid["exit_rank_below"],
        grid["max_hold"],
        grid["invested_ratio"],
    ):
        profile = Profile(*values)
        daily = simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
        selection = core.metrics(daily, "20220606", "20241231")
        confirmation = core.metrics(daily, "20250101", "20251231")
        full = core.metrics(daily, "20220606", "20251231")
        rows.append({"case_id": profile.case_id, **asdict(profile), **{f"selection_{k}": v for k, v in selection.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, **{f"full_{k}": v for k, v in full.items()}, "min_period_sharpe": min(selection["sharpe"], confirmation["sharpe"])})
    result = pd.DataFrame(rows)
    gate = protocol["selection_gate"]
    result["eligible"] = (
        (result.selection_cumulative_return > 0)
        & (result.confirmation_cumulative_return > 0)
        & (result.full_max_drawdown <= float(gate["full_max_drawdown_max"]))
        & (result.full_trades >= int(gate["full_trades_min"]))
    )
    result = result.sort_values(["eligible", "min_period_sharpe", "full_sharpe", "full_max_drawdown", "full_turnover", "case_id"], ascending=[False, False, False, True, True, True])
    result.to_csv(GRID_PATH, index=False, encoding="utf-8-sig")
    frozen = result[result.eligible].head(int(gate["promote_for_juejin"]))
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "grid_sha256": digest(GRID_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    actions_out, seen = [], set()
    for item in payload["profiles"]:
        profile = Profile(**{key: item[key] for key in Profile.__dataclass_fields__})
        _, actions = simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231", record_actions=True)
        action_key = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_key in seen:
            continue
        seen.add(action_key)
        path = ACTION_DIR / f"{profile.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        actions_out.append({"case_id": profile.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": actions_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only", "grid_cases": len(result), "eligible_cases": int(result.eligible.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(actions_out), "best": frozen.head(1).to_dict("records"), "validation_not_opened": True, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
