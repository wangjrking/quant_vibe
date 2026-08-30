from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_yisi_gap_backtest_20260701"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
LIMIT_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l1_raw_tables" / "limit_list_data.duckdb"
JUEJIN_STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")


@dataclass(frozen=True)
class Config:
    strategy_id: str = "research_yisi_gap_v0_20260701"
    start: str = "20220606"
    end: str = "20260630"
    lookback_days: int = 126
    min_limit_count_6m: int = 2
    min_stage_rise_6m: float = 0.40
    min_amount_20: float = 90000.0
    min_turnover_20: float = 1.0
    min_gap_pct: float = 0.005
    max_current_gap_distance: float = 0.20
    max_residual_gap: float = 0.03
    gap_break_tolerance: float = 0.005
    retest_zone: float = 0.03
    recent_window: int = 3
    top_n: int = 3
    holding_days: int = 5
    target_total_pct: float = 0.90
    slippage_ratio: float = 0.0015
    initial_cash: float = 600000.0


def to_gm_symbol(stock_code: str) -> str:
    code = str(stock_code).upper()
    if code.endswith(".SH"):
        return f"SHSE.{code[:6]}"
    if code.endswith(".SZ"):
        return f"SZSE.{code[:6]}"
    raise ValueError(f"unsupported stock_code={stock_code}")


def board_name(stock_code: str) -> str:
    code = str(stock_code).upper()
    if code.startswith("688") and code.endswith(".SH"):
        return "科创板"
    if code.startswith(("300", "301")) and code.endswith(".SZ"):
        return "创业板"
    return "主板"


def load_market(cfg: Config) -> pd.DataFrame:
    query = """
        SELECT
            stock_code, trade_date, name, market, list_date,
            open_qfq, high_qfq, low_qfq, close_qfq, pre_close_qfq,
            open, high, low, close, pre_close,
            pct_chg, amount, turnover_rate, turnover_rate_f,
            total_mv, circ_mv, ST_TYPE, ST_TYPE_name
        FROM STOCK_DAILY_DATA
        WHERE trade_date >= ? AND trade_date <= ?
          AND stock_code NOT LIKE '%.BJ'
          AND open_qfq IS NOT NULL
          AND high_qfq IS NOT NULL
          AND low_qfq IS NOT NULL
          AND close_qfq IS NOT NULL
          AND pre_close_qfq IS NOT NULL
    """
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        df = con.execute(query, [cfg.start, cfg.end]).fetchdf()
    finally:
        con.close()
    df["trade_date"] = df["trade_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    return df


def load_limit_counts(cfg: Config, calendar: list[str]) -> pd.DataFrame:
    con = duckdb.connect(str(LIMIT_DB), read_only=True)
    try:
        raw = con.execute(
            """
            SELECT trade_date, ts_code AS stock_code
            FROM limit_list_data
            WHERE trade_date >= ? AND trade_date <= ?
              AND ts_code NOT LIKE '%.BJ'
              AND "limit" = 'U'
            """,
            [cfg.start, cfg.end],
        ).fetchdf()
    finally:
        con.close()
    if raw.empty:
        return pd.DataFrame(columns=["stock_code", "trade_date", "limit_count_6m"])
    raw["trade_date"] = raw["trade_date"].astype(str)
    raw["stock_code"] = raw["stock_code"].astype(str)
    cal_index = {d: i for i, d in enumerate(calendar)}
    raw = raw[raw["trade_date"].isin(cal_index)]
    raw["date_idx"] = raw["trade_date"].map(cal_index)
    records = []
    for stock_code, g in raw.groupby("stock_code", sort=False):
        indices = sorted(g["date_idx"].tolist())
        left = 0
        for right, idx in enumerate(indices):
            while indices[left] < idx - cfg.lookback_days + 1:
                left += 1
            records.append((stock_code, calendar[idx], right - left + 1))
    return pd.DataFrame(records, columns=["stock_code", "trade_date", "limit_count_6m"])


def add_rolling_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = df.groupby("stock_code", sort=False)
    df["prev_high_qfq"] = g["high_qfq"].shift(1)
    df["prev_close_qfq"] = g["close_qfq"].shift(1)
    df["roll_min_close_6m"] = g["close_qfq"].transform(lambda s: s.rolling(cfg.lookback_days, min_periods=20).min())
    df["roll_max_close_6m"] = g["close_qfq"].transform(lambda s: s.rolling(cfg.lookback_days, min_periods=20).max())
    df["stage_rise_6m"] = df["roll_max_close_6m"] / df["roll_min_close_6m"] - 1.0
    df["amount_med_20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).median())
    df["turnover_med_20"] = g["turnover_rate"].transform(lambda s: s.rolling(20, min_periods=10).median())
    df["range_qfq"] = (df["high_qfq"] - df["low_qfq"]).abs()
    df["body_qfq"] = (df["close_qfq"] - df["open_qfq"]).abs()
    df["body_range_ratio"] = df["body_qfq"] / df["range_qfq"].replace(0, pd.NA)
    df["doji_like"] = (df["body_range_ratio"] <= 0.25) | (df["body_qfq"] / df["close_qfq"] <= 0.015)
    df["recent_doji_count"] = g["doji_like"].transform(lambda s: s.rolling(cfg.recent_window, min_periods=1).sum())
    df["recent_low_min"] = g["low_qfq"].transform(lambda s: s.rolling(cfg.recent_window, min_periods=1).min())
    return df


def is_st_row(row: pd.Series) -> bool:
    st_type = str(row.get("ST_TYPE") or "").strip()
    st_type_name = str(row.get("ST_TYPE_name") or "").strip()
    name = str(row.get("name") or "").strip().upper()
    if st_type and st_type not in {"0", "None", "nan"}:
        return True
    if st_type_name and st_type_name not in {"0", "None", "nan"}:
        return True
    return name.startswith("ST") or name.startswith("*ST")


def add_hard_gate_flags(df: pd.DataFrame) -> pd.DataFrame:
    st_type = df["ST_TYPE"].fillna("").astype(str).str.strip()
    st_type_name = df["ST_TYPE_name"].fillna("").astype(str).str.strip()
    name = df["name"].fillna("").astype(str).str.strip().str.upper()
    df["is_st_flag"] = (
        ((st_type != "") & (~st_type.isin(["0", "None", "nan"])))
        | ((st_type_name != "") & (~st_type_name.isin(["0", "None", "nan"])))
        | name.str.startswith("ST")
        | name.str.startswith("*ST")
    )
    board_limit = pd.Series(0.10, index=df.index)
    board_limit.loc[df["stock_code"].astype(str).str.startswith(("300", "301", "688"))] = 0.20
    open_ratio = df["open_qfq"] / df["pre_close_qfq"] - 1.0
    df["is_open_limit_up_flag"] = open_ratio >= board_limit - 0.002
    df["is_open_limit_down_flag"] = open_ratio <= -board_limit + 0.002
    return df


def is_open_limit_up(row: pd.Series) -> bool:
    pre_close = float(row["pre_close_qfq"])
    open_price = float(row["open_qfq"])
    if pre_close <= 0:
        return False
    limit = 0.20 if str(row["stock_code"]).startswith(("300", "301", "688")) else 0.10
    return open_price / pre_close - 1.0 >= limit - 0.002


def is_open_limit_down(row: pd.Series) -> bool:
    pre_close = float(row["pre_close_qfq"])
    open_price = float(row["open_qfq"])
    if pre_close <= 0:
        return False
    limit = -0.20 if str(row["stock_code"]).startswith(("300", "301", "688")) else -0.10
    return open_price / pre_close - 1.0 <= limit + 0.002


def build_signals(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    calendar = sorted(df["trade_date"].unique().tolist())
    next_date = {d: calendar[i + 1] for i, d in enumerate(calendar[:-1])}
    open_limit_up_pairs = set(
        df.loc[df["is_open_limit_up_flag"], ["trade_date", "stock_code"]].itertuples(index=False, name=None)
    )
    candidates = []
    filter_counts = {
        "rows": int(len(df)),
        "bj_excluded_by_query": True,
        "st_excluded": 0,
        "activity_fail": 0,
        "liquidity_fail": 0,
        "gap_candidates": 0,
        "distance_fail": 0,
        "stability_fail": 0,
        "limit_up_buy_fail": 0,
    }

    for stock_code, g in df.groupby("stock_code", sort=False):
        active_gaps: list[dict] = []
        rows = list(g.itertuples(index=False))
        for row in rows:
            rowd = row._asdict()
            trade_date = rowd["trade_date"]
            if trade_date not in next_date:
                continue
            if pd.notna(rowd["prev_high_qfq"]) and rowd["low_qfq"] > rowd["prev_high_qfq"] * (1.0 + cfg.min_gap_pct):
                active_gaps.append(
                    {
                        "gap_date": trade_date,
                        "gap_lower": float(rowd["prev_high_qfq"]),
                        "gap_upper": float(rowd["low_qfq"]),
                        "gap_pct": float(rowd["low_qfq"] / rowd["prev_high_qfq"] - 1.0),
                        "invalid": False,
                    }
                )
            if not active_gaps:
                continue
            if rowd.get("is_st_flag"):
                filter_counts["st_excluded"] += 1
                continue
            if rowd.get("limit_count_6m", 0) < cfg.min_limit_count_6m and (
                pd.isna(rowd.get("stage_rise_6m")) or rowd["stage_rise_6m"] < cfg.min_stage_rise_6m
            ):
                filter_counts["activity_fail"] += 1
                continue
            if (
                pd.isna(rowd.get("amount_med_20"))
                or pd.isna(rowd.get("turnover_med_20"))
                or rowd["amount_med_20"] < cfg.min_amount_20
                or rowd["turnover_med_20"] < cfg.min_turnover_20
            ):
                filter_counts["liquidity_fail"] += 1
                continue

            for gap in list(active_gaps):
                if gap["invalid"] or trade_date <= gap["gap_date"]:
                    continue
                lower = gap["gap_lower"]
                if rowd["close_qfq"] < lower * (1.0 - cfg.gap_break_tolerance):
                    gap["invalid"] = True
                    continue
                filter_counts["gap_candidates"] += 1
                if rowd["low_qfq"] > lower * (1.0 + cfg.retest_zone):
                    continue
                if rowd["low_qfq"] <= lower and rowd["close_qfq"] > lower:
                    residual_gap = float(rowd["close_qfq"] / lower - 1.0)
                elif rowd["low_qfq"] > lower:
                    residual_gap = float(rowd["low_qfq"] / lower - 1.0)
                else:
                    continue
                current_distance = float(rowd["close_qfq"] / lower - 1.0)
                if (
                    residual_gap < 0
                    or residual_gap > cfg.max_residual_gap
                    or current_distance < 0
                    or current_distance > cfg.max_current_gap_distance
                ):
                    filter_counts["distance_fail"] += 1
                    continue
                if rowd["recent_doji_count"] < 1 or rowd["recent_low_min"] < lower * (1.0 - cfg.gap_break_tolerance):
                    filter_counts["stability_fail"] += 1
                    continue
                buy_date = next_date[trade_date]
                if (buy_date, stock_code) in open_limit_up_pairs:
                    filter_counts["limit_up_buy_fail"] += 1
                    continue
                score = (
                    -current_distance * 4.0
                    - residual_gap * 6.0
                    + min(float(rowd.get("limit_count_6m") or 0), 5.0) * 0.04
                    + min(float(rowd.get("stage_rise_6m") or 0), 2.0) * 0.08
                    + min(float(rowd.get("turnover_med_20") or 0), 10.0) * 0.005
                )
                candidates.append(
                    {
                        "signal_date": trade_date,
                        "buy_date": buy_date,
                        "stock_code": stock_code,
                        "symbol": to_gm_symbol(stock_code),
                        "name": rowd["name"],
                        "board": board_name(stock_code),
                        "gap_date": gap["gap_date"],
                        "gap_retest_date": trade_date,
                        "gap_lower_qfq": lower,
                        "gap_upper_qfq": gap["gap_upper"],
                        "gap_pct": gap["gap_pct"],
                        "residual_gap_pct": residual_gap,
                        "current_gap_distance_pct": current_distance,
                        "limit_count_6m": int(rowd.get("limit_count_6m") or 0),
                        "stage_rise_6m": float(rowd.get("stage_rise_6m") or 0),
                        "amount_med_20": float(rowd.get("amount_med_20") or 0),
                        "turnover_med_20": float(rowd.get("turnover_med_20") or 0),
                        "score": score,
                    }
                )

    cand = pd.DataFrame(candidates)
    if cand.empty:
        return cand, filter_counts
    cand = cand.sort_values(["signal_date", "score"], ascending=[True, False])
    cand["rank"] = cand.groupby("signal_date").cumcount() + 1
    signals = cand[cand["rank"] <= cfg.top_n].copy()
    signals["target_weight"] = cfg.target_total_pct / cfg.top_n
    return signals.reset_index(drop=True), filter_counts


def run_local_backtest(df: pd.DataFrame, signals: pd.DataFrame, cfg: Config) -> dict:
    if signals.empty:
        return {"error": "no_signals"}
    px = df.set_index(["trade_date", "stock_code"]).sort_index()
    calendar = sorted(df["trade_date"].unique().tolist())
    idx = {d: i for i, d in enumerate(calendar)}
    by_buy = {d: g.to_dict("records") for d, g in signals.groupby("buy_date")}
    cash = cfg.initial_cash
    positions: list[dict] = []
    equity_curve = []
    trades = []
    for date in calendar:
        if date < signals["buy_date"].min():
            continue
        # sell first
        remaining = []
        for pos in positions:
            due = idx[date] >= pos["sell_idx"]
            row = px.loc[(date, pos["stock_code"])] if (date, pos["stock_code"]) in px.index else None
            if due and row is not None and not bool(row.get("is_open_limit_down_flag", False)):
                sell_price = float(row["open_qfq"]) * (1.0 - cfg.slippage_ratio)
                cash += pos["shares"] * sell_price
                ret = sell_price / pos["buy_price"] - 1.0
                trades.append({**pos, "sell_date": date, "sell_price": sell_price, "return": ret})
            else:
                remaining.append(pos)
        positions = remaining
        # buy
        for sig in by_buy.get(date, []):
            if len(positions) >= cfg.top_n:
                break
            stock_code = sig["stock_code"]
            if (date, stock_code) not in px.index:
                continue
            row = px.loc[(date, stock_code)]
            if bool(row.get("is_open_limit_up_flag", False)) or bool(row.get("is_st_flag", False)):
                continue
            target_value = cfg.initial_cash * (cfg.target_total_pct / cfg.top_n)
            buy_value = min(cash, target_value)
            if buy_value <= 0:
                continue
            buy_price = float(row["open_qfq"]) * (1.0 + cfg.slippage_ratio)
            shares = buy_value / buy_price
            cash -= shares * buy_price
            positions.append(
                {
                    "stock_code": stock_code,
                    "name": sig["name"],
                    "buy_date": date,
                    "buy_price": buy_price,
                    "shares": shares,
                    "sell_idx": min(idx[date] + cfg.holding_days, len(calendar) - 1),
                    "signal_date": sig["signal_date"],
                    "gap_date": sig["gap_date"],
                    "gap_retest_date": sig["gap_retest_date"],
                }
            )
        value = cash
        for pos in positions:
            if (date, pos["stock_code"]) in px.index:
                value += pos["shares"] * float(px.loc[(date, pos["stock_code"])]["close_qfq"])
        equity_curve.append({"trade_date": date, "equity": value, "positions": len(positions)})

    curve = pd.DataFrame(equity_curve)
    if curve.empty:
        return {"error": "empty_curve"}
    curve["ret"] = curve["equity"].pct_change().fillna(0.0)
    years = max(len(curve) / 252.0, 1e-9)
    total_return = curve["equity"].iloc[-1] / cfg.initial_cash - 1.0
    annual_return = (curve["equity"].iloc[-1] / cfg.initial_cash) ** (1.0 / years) - 1.0
    sharpe = 0.0
    if curve["ret"].std(ddof=0) > 0:
        sharpe = float(curve["ret"].mean() / curve["ret"].std(ddof=0) * math.sqrt(252))
    drawdown = curve["equity"] / curve["equity"].cummax() - 1.0
    trades_df = pd.DataFrame(trades)
    summary = {
        "start": str(curve["trade_date"].iloc[0]),
        "end": str(curve["trade_date"].iloc[-1]),
        "days": int(len(curve)),
        "initial_cash": cfg.initial_cash,
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "trade_count": int(len(trades_df)),
        "win_rate": float((trades_df["return"] > 0).mean()) if not trades_df.empty else None,
        "avg_positions": float(curve["positions"].mean()),
        "signal_rows": int(len(signals)),
        "buy_days": int(signals["buy_date"].nunique()),
    }
    curve.to_csv(REPORT_DIR / "local_equity_curve.csv", index=False, encoding="utf-8-sig")
    trades_df.to_csv(REPORT_DIR / "local_trades.csv", index=False, encoding="utf-8-sig")
    return summary


def run_juejin(signals_path: Path, cfg: Config) -> dict:
    log_file = REPORT_DIR / "juejin_yisi_gap_top3_h5.log"
    if not (JUEJIN_STRATEGY_DIR / "main.py").exists():
        return {"returncode": None, "error": "juejin_strategy_main_missing", "log_file": str(log_file)}
    env = {
        **dict(),
    }
    cmd = [
        sys.executable,
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(JUEJIN_STRATEGY_DIR),
        "--signal-file",
        str(signals_path),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(cfg.top_n),
        "--holding-days",
        str(cfg.holding_days),
        "--target-position-pct",
        str(cfg.target_total_pct / cfg.top_n),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        str(cfg.initial_cash),
        "--backtest-slippage-ratio",
        str(cfg.slippage_ratio),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(MAIN),
        text=True,
        capture_output=True,
        env={
            **__import__("os").environ,
            "GM_SCORE_DB": "",
            "GM_SCORE_TABLE": "",
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_INTRADAY_RISK_MODE": "0",
            "GM_MARKET_DB": str(L2_DB),
            "GM_DB_IMMUTABLE_READ": "0",
        },
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "log_file": str(log_file),
    }


def write_report(cfg: Config, signals: pd.DataFrame, filters: dict, local_summary: dict, juejin_summary: dict) -> None:
    lines = [
        "# 一丝缺口选股法研究回测结果",
        "",
        "## 口径说明",
        "",
        "- 策略名称：一丝缺口选股法。",
        "- 资产层级：research-only，不是生产策略。",
        "- 数据输入：`l2_stock_daily_data.duckdb::STOCK_DAILY_DATA` 与 `limit_list_data.duckdb::limit_list_data`。",
        "- 价格形态字段：显式使用 `open_qfq/high_qfq/low_qfq/close_qfq/pre_close_qfq`。",
        "- 股票池：排除 `.BJ`，排除 ST/风险警示，跳过次日开盘涨停买入。",
        "- 本轮为固定参数原型回测，不做参数搜索。",
        "",
        "## 参数",
        "",
        "```json",
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 信号概况",
        "",
        f"- 信号行数：{len(signals)}",
        f"- 买入日数量：{signals['buy_date'].nunique() if not signals.empty else 0}",
        f"- 覆盖股票数：{signals['stock_code'].nunique() if not signals.empty else 0}",
        f"- 最新信号日：{signals['signal_date'].max() if not signals.empty else ''}",
        f"- 最新买入日：{signals['buy_date'].max() if not signals.empty else ''}",
        "",
        "## 本地可复现回测",
        "",
        "```json",
        json.dumps(local_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 掘金回测尝试",
        "",
        "```json",
        json.dumps(juejin_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 过滤统计",
        "",
        "```json",
        json.dumps(filters, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 结论",
        "",
        "本报告只用于判断策略原型是否值得继续调参。若掘金回测失败或与本地结果不一致，以掘金可复现回测为正式收益口径；本地结果不能直接发布到 L8。",
        "",
    ]
    (REPORT_DIR / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg = Config()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_market(cfg)
    calendar = sorted(df["trade_date"].unique().tolist())
    limit_counts = load_limit_counts(cfg, calendar)
    df = add_rolling_features(df, cfg)
    df = add_hard_gate_flags(df)
    if not limit_counts.empty:
        df = df.merge(limit_counts, on=["stock_code", "trade_date"], how="left")
    else:
        df["limit_count_6m"] = 0
    df["limit_count_6m"] = df["limit_count_6m"].fillna(0).astype(int)
    signals, filters = build_signals(df, cfg)
    signals_path = REPORT_DIR / "yisi_gap_signals_top3_h5.csv"
    signals.to_csv(signals_path, index=False, encoding="utf-8-sig")
    local_summary = run_local_backtest(df, signals, cfg)
    juejin_summary = run_juejin(signals_path, cfg) if not signals.empty else {"error": "no_signals"}
    summary = {
        "config": asdict(cfg),
        "signals_path": str(signals_path),
        "local_summary": local_summary,
        "juejin_summary": juejin_summary,
        "filters": filters,
        "data_rows": int(len(df)),
        "data_min_date": str(df["trade_date"].min()),
        "data_max_date": str(df["trade_date"].max()),
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(cfg, signals, filters, local_summary, juejin_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
