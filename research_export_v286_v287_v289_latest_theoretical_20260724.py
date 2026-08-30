from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v260_weak_breadth_v286_20260723 as v286
import research_active_l4_v286_pareto_breadth_v289_20260723 as v289
import research_active_l4_v286_weak_threshold_v287_20260723 as v287
import research_active_l4_v258_position_warmup_v260_20260723 as v260


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "quant/data_file"
L2 = DATA / "production_assets/duckdb/l2_stock_daily_data.duckdb"
OUT = DATA / "reports/strategy_agent_v286_v287_v289_latest_theoretical_20260724"

CASES = {
    "V286": (
        v286,
        DATA / "reports/strategy_agent_active_l4_v260_weak_breadth_v286_20260723",
    ),
    "V287": (
        v287,
        DATA / "reports/strategy_agent_active_l4_v286_weak_threshold_v287_20260723",
    ),
    "V289": (
        v289,
        DATA / "reports/strategy_agent_active_l4_v286_pareto_breadth_v289_20260723",
    ),
}


def next_weekday(value: str) -> str:
    day = datetime.strptime(value, "%Y%m%d").date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.strftime("%Y%m%d")


def active_holdings(actions: pd.DataFrame, dates: np.ndarray) -> dict[str, int]:
    date_index = {str(value): idx for idx, value in enumerate(dates.astype(str))}
    holdings: dict[str, int] = {}
    for row in actions.itertuples(index=False):
        stock = str(row.stock_code)
        if str(row.action).upper() == "BUY":
            holdings[stock] = date_index[str(row.signal_date)]
        else:
            holdings.pop(stock, None)
    return holdings


def selected_case(base: Path) -> tuple[dict, dict]:
    protocol = json.loads((base / "preregistered_protocol.json").read_text(encoding="utf-8"))
    selected = json.loads(
        (base / "selected_candidate_before_known_2026.json").read_text(encoding="utf-8")
    )
    return protocol, selected["selected_definition"]


def entry_slots(label: str, arrays: dict, score: np.ndarray, definition: dict, protocol: dict) -> np.ndarray:
    if label == "V286":
        return v286.entry_slots_schedule(arrays, definition, protocol)
    if label == "V287":
        return v287.entry_slots(arrays, definition)
    return v289.v288.entry_slots(arrays, score, definition, protocol)


def signal_rows(
    label: str,
    module,
    arrays: dict,
    score: np.ndarray,
    order: np.ndarray,
    protocol: dict,
    definition: dict,
    actions: pd.DataFrame,
) -> list[dict]:
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    stock_index = {stock: idx for idx, stock in enumerate(stocks)}
    t = len(dates) - 1
    signal_date = str(dates[t])
    buy_date = next_weekday(signal_date)
    holdings = active_holdings(actions, dates)

    selection = v260.v258.v252.v174.selection_mask(
        arrays, definition["max_rank_deterioration"]
    )[t]
    clean = arrays["signal_clean"][t] & np.isfinite(score[t]) & selection
    clean &= arrays["listed_days"][t] >= int(protocol["fixed_universe"]["listed_days_min"])
    clean &= np.isfinite(arrays["amount"][t])
    clean &= arrays["amount"][t] >= int(protocol["fixed_universe"]["amount_min"])
    clean &= np.isfinite(arrays["total_mv"][t])
    clean &= arrays["total_mv"][t] >= int(protocol["fixed_universe"]["mv_min"])
    clean &= np.isfinite(arrays["turnover_rate"][t])
    clean &= arrays["turnover_rate"][t] >= 0
    clean &= arrays["turnover_rate"][t] <= float(protocol["fixed_universe"]["turnover_max"])
    ranked = [int(idx) for idx in order[t] if clean[int(idx)]]
    best_unheld = max(
        (float(score[t, idx]) for idx in ranked if stocks[idx] not in holdings),
        default=-np.inf,
    )
    min_hold = v260.v258.v252.v162.min_hold_schedule(
        arrays, definition["min_hold_policy"]
    )
    planned_sells: dict[str, str] = {}
    for stock, entry_t in sorted(holdings.items()):
        idx = stock_index[stock]
        age = t - entry_t
        current = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
        score_exit = (
            age >= int(min_hold[t])
            and current < float(definition["sell_score_below"])
            and best_unheld - current >= float(definition["replacement_advantage"])
        )
        if age >= int(definition["max_hold_days"]):
            planned_sells[stock] = "达到最长持有期"
        elif score_exit:
            planned_sells[stock] = "持仓评分落后可替代候选"

    holdings_after_sell = set(holdings) - set(planned_sells)
    max_positions = int(v260.position_schedule(arrays, definition, None)[t])
    daily_slots = int(entry_slots(label, arrays, score, definition, protocol)[t])
    slots = min(daily_slots, max(max_positions - len(holdings_after_sell), 0))
    buys = [
        idx for idx in ranked if stocks[idx] not in holdings_after_sell
    ][:slots]

    target = v260.v258.v252.v162.v153.schedules(arrays, definition)[1]
    multiplier = v260.v258.v252.warmup_multiplier(
        arrays, score, definition, protocol, None
    )
    rank_map = {
        int(idx): rank
        for rank, idx in enumerate(order[t], start=1)
        if np.isfinite(score[t, int(idx)])
    }
    denominator = int(np.isfinite(score[t]).sum())

    with duckdb.connect(str(L2), read_only=True) as con:
        names = con.execute(
            """
            SELECT stock_code, any_value(name) AS name
            FROM STOCK_DAILY_DATA
            WHERE trade_date=?
            GROUP BY stock_code
            """,
            [signal_date],
        ).fetchdf().set_index("stock_code")

    def base(stock: str) -> dict:
        idx = stock_index[stock]
        return {
            "strategy": label,
            "signal_date": signal_date,
            "buy_date": buy_date,
            "stock_code": stock,
            "name": str(names.loc[stock, "name"]) if stock in names.index else "",
            "score_10d_smoothed": (
                float(score[t, idx]) if np.isfinite(score[t, idx]) else None
            ),
            "score_rank": rank_map.get(idx),
            "score_denominator": denominator,
            "buy_day_hard_gate_complete": False,
            "status": "research_only_pending_buy_day_hard_gate",
        }

    rows = []
    for stock, reason in planned_sells.items():
        rows.append(
            {
                **base(stock),
                "action": "SELL",
                "target_pct": 0.0,
                "reason": reason,
            }
        )
    split = max(len(buys), 1)
    for idx in buys:
        stock = stocks[idx]
        rows.append(
            {
                **base(stock),
                "action": "BUY",
                "target_pct": float(target[t] * multiplier[t, idx] / split),
                "reason": f"当日有效候选前{daily_slots}名，补足目标持仓",
            }
        )
    for stock in sorted(holdings_after_sell):
        rows.append(
            {
                **base(stock),
                "action": "HOLD",
                "target_pct": None,
                "reason": "未触发最长持有期或评分替换卖出条件",
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = v260.v258.v252.fresh_arrays()
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    dates = arrays["dates"].astype(str)
    previous_signal_date = str(dates[-2])
    all_rows = []
    summaries = []
    for label, (module, base) in CASES.items():
        protocol, definition = selected_case(base)
        runner = module.run_case if label != "V289" else module.v288.run_case
        _, actions = runner(
            arrays,
            score,
            order,
            definition,
            protocol,
            previous_signal_date,
            record_actions=True,
        )
        rows = signal_rows(
            label, module, arrays, score, order, protocol, definition, actions
        )
        all_rows.extend(rows)
        summaries.append(
            {
                "strategy": label,
                "signal_date": str(dates[-1]),
                "buy_date": next_weekday(str(dates[-1])),
                "buy_count": sum(row["action"] == "BUY" for row in rows),
                "sell_count": sum(row["action"] == "SELL" for row in rows),
                "hold_count": sum(row["action"] == "HOLD" for row in rows),
            }
        )
    frame = pd.DataFrame(all_rows)
    frame.to_csv(OUT / "latest_theoretical_signals.csv", index=False, encoding="utf-8-sig")
    report = {
        "schema_version": 1,
        "status": "research_only_not_l5_l6",
        "input_contract": "approved formal 1D/3D/5D/10D key intersection; only 10D score",
        "signal_date": str(dates[-1]),
        "buy_date": next_weekday(str(dates[-1])),
        "buy_day_market_available": False,
        "buy_day_hard_gate_complete": False,
        "l7_execution_allowed": False,
        "summaries": summaries,
        "signal_file": str(OUT / "latest_theoretical_signals.csv"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
