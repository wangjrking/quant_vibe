from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_next_open_frequency_smooth_20260719"
SIGNAL_FILE = REPORT_DIR / "signals" / "robust_center_cap90.csv"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT_CSV = REPORT_DIR / "independent_sell_frequency_local_screen.csv"
OUT_JSON = REPORT_DIR / "independent_sell_frequency_local_screen.json"

SLIPPAGE = 0.003
MAX_POSITIONS = 2


CASES = [
    {"name": f"min{min_hold}_max{max_hold}_r{ratio:.3f}_e{edge:.3f}", "min_hold": min_hold, "max_hold": max_hold, "ratio": ratio, "edge": edge}
    for min_hold in [1, 2]
    for max_hold in [2, 3, 4, 5, 6]
    if max_hold > min_hold
    for ratio in [0.95, 0.98, 1.00, 1.02, 1.05]
    for edge in [0.0, 0.005, 0.01]
]


def load_inputs() -> tuple[pd.DataFrame, list[str], dict, dict]:
    signal = pd.read_csv(
        SIGNAL_FILE,
        encoding="utf-8-sig",
        dtype={"signal_date": str, "buy_date": str, "stock_code": str},
    )
    signal = signal[~signal["stock_code"].str.endswith(".BJ", na=False)].copy()
    signal["pred_prob"] = pd.to_numeric(signal["pred_prob"], errors="coerce")
    signal["target_pct"] = pd.to_numeric(signal["target_pct"], errors="coerce").fillna(0.0)
    signal = signal.dropna(subset=["pred_prob"]).sort_values(["buy_date", "pred_prob"], ascending=[True, False])
    codes = sorted(signal["stock_code"].unique().tolist())

    market = duckdb.connect(str(MARKET_DB), read_only=True)
    market.execute("create temp table need_codes(stock_code varchar)")
    market.executemany("insert into need_codes values (?)", [(code,) for code in codes])
    prices = market.execute(
        """
        select cast(trade_date as varchar) trade_date, stock_code, open
        from STOCK_DAILY_DATA
        where stock_code in (select stock_code from need_codes)
          and trade_date >= ?
        """,
        [str(signal["buy_date"].min())],
    ).fetchdf()
    dates = [
        str(row[0])
        for row in market.execute(
            "select distinct trade_date from STOCK_DAILY_DATA where trade_date >= ? order by trade_date",
            [str(signal["buy_date"].min())],
        ).fetchall()
    ]
    market.close()
    open_map = {
        (str(row.trade_date), str(row.stock_code)): float(row.open)
        for row in prices.itertuples()
        if pd.notna(row.open) and float(row.open) > 0
    }

    score = duckdb.connect(str(SCORE_DB), read_only=True)
    score.execute("create temp table need_codes(stock_code varchar)")
    score.executemany("insert into need_codes values (?)", [(code,) for code in codes])
    scores = score.execute(
        f"""
        select trade_date, stock_code, pred_prob
        from {SCORE_TABLE}
        where stock_code in (select stock_code from need_codes)
          and trade_date >= ?
        """,
        [str(signal["signal_date"].min())],
    ).fetchdf()
    score.close()
    score_map = {
        (str(row.trade_date), str(row.stock_code)): float(row.pred_prob)
        for row in scores.itertuples()
        if pd.notna(row.pred_prob)
    }
    return signal, dates, open_map, score_map


def period_stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, mdd


def evaluate(signal: pd.DataFrame, dates: list[str], open_map: dict, score_map: dict, case: dict) -> dict:
    date_index = {date: index for index, date in enumerate(dates)}
    signals = {date: day.to_dict("records") for date, day in signal.groupby("buy_date")}
    decision_date = {date: str(day["signal_date"].iloc[0]) for date, day in signal.groupby("buy_date")}
    positions: dict[str, dict] = {}
    daily_return = []
    exposure = []
    buy_count = 0
    sell_count = 0
    replace_count = 0

    for index, today in enumerate(dates):
        day_return = 0.0
        if index > 0:
            yesterday = dates[index - 1]
            for code, state in positions.items():
                previous_open = open_map.get((yesterday, code))
                current_open = open_map.get((today, code))
                if previous_open and current_open:
                    day_return += state["weight"] * (current_open / previous_open - 1.0)

        today_signals = signals.get(today, [])
        score_date = decision_date.get(today)
        candidate_scores = [float(item["pred_prob"]) for item in today_signals if pd.notna(item.get("pred_prob"))]
        best_candidate_score = max(candidate_scores) if candidate_scores else None

        sold_weight = 0.0
        for code in list(positions):
            state = positions[code]
            age = index - state["entry_index"]
            current_score = score_map.get((score_date, code)) if score_date else None
            replace = (
                age >= int(case["min_hold"])
                and best_candidate_score is not None
                and current_score is not None
                and best_candidate_score >= current_score * float(case["ratio"])
                and best_candidate_score - current_score >= float(case["edge"])
            )
            timeout = age >= int(case["max_hold"])
            if replace or timeout:
                sold_weight += float(state["weight"])
                del positions[code]
                sell_count += 1
                replace_count += int(replace)

        bought_weight = 0.0
        for item in today_signals:
            code = str(item["stock_code"])
            if code in positions or len(positions) >= MAX_POSITIONS:
                continue
            weight = max(float(item["target_pct"]), 0.0)
            if weight <= 0:
                continue
            positions[code] = {"weight": weight, "entry_index": index}
            bought_weight += weight
            buy_count += 1

        day_return -= (sold_weight + bought_weight) * SLIPPAGE
        daily_return.append(day_return)
        exposure.append(sum(float(state["weight"]) for state in positions.values()))

    daily = pd.Series(daily_return, index=dates, dtype=float)
    annual, sharpe, mdd = period_stats(daily)
    dated = daily.copy()
    dated.index = pd.to_datetime(dated.index)
    yearly = [period_stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 40]
    rolling = [period_stats(daily.iloc[start : start + 60]) for start in range(0, max(len(daily) - 59, 1), 20)]
    return {
        **case,
        "local_annual": annual,
        "local_sharpe": sharpe,
        "local_mdd": mdd,
        "worst_year_annual": min((item[0] for item in yearly), default=0.0),
        "worst_year_sharpe": min((item[1] for item in yearly), default=0.0),
        "rolling60_min_annual": min((item[0] for item in rolling), default=0.0),
        "rolling60_min_sharpe": min((item[1] for item in rolling), default=0.0),
        "rolling60_median_sharpe": float(np.median([item[1] for item in rolling])) if rolling else 0.0,
        "mean_exposure": float(np.mean(exposure)),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "replace_count": replace_count,
    }


def main() -> None:
    signal, dates, open_map, score_map = load_inputs()
    rows = [evaluate(signal, dates, open_map, score_map, case) for case in CASES]
    frame = pd.DataFrame(rows)
    frame["stability_score"] = (
        frame["local_sharpe"]
        + 0.40 * frame["worst_year_sharpe"]
        + 0.25 * frame["rolling60_median_sharpe"]
        + 0.10 * frame["rolling60_min_sharpe"]
    )
    frame = frame.sort_values(["stability_score", "local_sharpe", "local_annual"], ascending=False)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(
            {
                "status": "research_only_local_prescreen_not_juejin",
                "signal_file": str(SIGNAL_FILE),
                "score_db": str(SCORE_DB),
                "score_table": SCORE_TABLE,
                "market_db": str(MARKET_DB),
                "execution_price": "raw_unadjusted_open",
                "factor_model_price_semantics": "qfq_explicit_upstream",
                "slippage_each_side": SLIPPAGE,
                "sell_rule": "current holding score versus current new-candidate score; no entry-score dependency",
                "rows": frame.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.head(20).to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
