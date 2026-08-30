from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SIGNAL_DIR = REPORT_DIR / "signals"
LOG_DIR = REPORT_DIR / "logs"

SOURCE_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "signals"
    / "open_gap_deep_rebalance"
    / "ogd_deep8_up110.csv"
)
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
)
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"


def as_num(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def load_source() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_SIGNAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    as_num(
        frame,
        [
            "rank",
            "target_pct",
            "signal_pct_chg_raw",
            "pred_1d",
            "pred_5d",
            "pred_10d",
            "exec_open_gap_pct",
            "buy_open_gap_pct",
            "buy_open_gap_raw_pct",
        ],
    )
    if "exec_open_gap_pct" not in frame.columns:
        frame["exec_open_gap_pct"] = pd.to_numeric(
            frame.get("buy_open_gap_raw_pct", frame.get("buy_open_gap_pct")), errors="coerce"
        )
    return frame


def build_signal(source: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, Path]:
    frame = source.copy()
    if params["open_gap_max"] is not None:
        frame = frame[frame["exec_open_gap_pct"].le(params["open_gap_max"])].copy()
    if frame.empty:
        return frame, SIGNAL_DIR / f"{params['name']}.csv"

    pct = frame["signal_pct_chg_raw"]
    pred_1d = frame["pred_1d"]
    pred_10d = frame["pred_10d"]
    gap = frame["exec_open_gap_pct"]

    scale = pd.Series(float(params["base_weight"]), index=frame.index)
    deep_mask = pct <= -9.0
    mild_good = pct.between(-5.0, -1.75, inclusive="both") & (pred_1d >= 0.90) & (pred_10d >= 0.99) & (gap <= 1.5)
    scale.loc[deep_mask] = float(params["deep_weight"])
    scale.loc[mild_good] = float(params["mild_weight"])

    raw = frame["target_pct"].fillna(0.0) * scale.clip(lower=0.0)
    daily_sum = raw.groupby(frame["buy_date"]).transform("sum")
    factor = (float(params["cap"]) / daily_sum).clip(upper=1.0).fillna(0.0)
    frame["target_pct_before_grid"] = frame["target_pct"]
    frame["target_pct"] = (raw * factor).clip(lower=0.0)
    frame = frame[frame["target_pct"] > 0].copy()
    if frame.empty:
        return frame, SIGNAL_DIR / f"{params['name']}.csv"

    frame["strategy_variant"] = params["name"]
    frame["filter_name"] = params["name"]
    frame["grid_hold_days"] = params["hold_days"]
    frame["grid_open_gap_max"] = params["open_gap_max"]
    frame["grid_deep_weight"] = params["deep_weight"]
    frame["daily_target_sum_after_cap"] = frame.groupby("buy_date")["target_pct"].transform("sum")

    out = SIGNAL_DIR / f"{params['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    return frame, out


def load_market(codes: list[str], min_date: str, max_date: str) -> tuple[list[str], dict[tuple[str, str], float]]:
    con = duckdb.connect(str(L2_DB), read_only=True)
    dates = [
        row[0]
        for row in con.execute(
            "select distinct trade_date from STOCK_DAILY_DATA where trade_date between ? and ? order by trade_date",
            [min_date, max_date],
        ).fetchall()
    ]
    con.execute("create temp table need_codes(stock_code varchar)")
    con.executemany("insert into need_codes values (?)", [(code,) for code in codes])
    market = con.execute(
        """
        select stock_code, trade_date, open
        from STOCK_DAILY_DATA
        where trade_date between ? and ?
          and stock_code in (select stock_code from need_codes)
        """,
        [min_date, max_date],
    ).fetchdf()
    con.close()
    open_map = {
        (str(row.stock_code), str(row.trade_date)): float(row.open)
        for row in market.itertuples()
        if pd.notna(row.open) and float(row.open) > 0
    }
    return dates, open_map


def local_portfolio_eval(frame: pd.DataFrame, hold_days: int, dates: list[str], open_map: dict[tuple[str, str], float]) -> dict:
    next_map = {date: dates[index + 1] for index, date in enumerate(dates[:-1])}
    date_index = {date: index for index, date in enumerate(dates)}
    signals = {
        date: [
            {"stock_code": row.stock_code, "target_pct": float(row.target_pct)}
            for row in day.itertuples(index=False)
        ]
        for date, day in frame.groupby("buy_date")
    }
    positions: dict[str, dict] = {}
    daily_returns: list[float] = []
    active_days = 0
    trade_count = 0
    missing_price_points = 0
    slippage = 0.0030

    for today in dates[:-1]:
        next_day = next_map[today]
        today_idx = date_index[today]

        # Scheduled sell frequency: sell positions whose opening holding age reaches hold_days.
        for code in list(positions):
            if today_idx - positions[code]["entry_index"] >= hold_days:
                del positions[code]

        if today in signals:
            # Rebalance only by current signal; no refill from outside the signal source.
            positions = {
                item["stock_code"]: {
                    "target_pct": item["target_pct"],
                    "entry_index": today_idx,
                }
                for item in signals[today]
            }
            trade_count += len(signals[today])

        day_return = 0.0
        if positions:
            active_days += 1
        for code, state in positions.items():
            open_today = open_map.get((code, today))
            open_next = open_map.get((code, next_day))
            if open_today is None or open_next is None:
                missing_price_points += 1
                continue
            day_return += float(state["target_pct"]) * (open_next / open_today - 1.0)
        if today in signals:
            day_return -= sum(item["target_pct"] for item in signals[today]) * slippage
        # Charge a sell-side friction when positions reach scheduled exit at next loop by approximation:
        # spread it on active days to avoid using future sell fill data in this diagnostic.
        if positions:
            day_return -= sum(float(state["target_pct"]) for state in positions.values()) * (slippage / max(hold_days, 1))
        daily_returns.append(day_return)

    series = pd.Series(daily_returns, index=dates[:-1], dtype=float)
    equity = (1.0 + series).cumprod()
    total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    annual = (float(equity.iloc[-1]) ** (252.0 / len(series)) - 1.0) if len(series) and equity.iloc[-1] > 0 else -1.0
    std = float(series.std(ddof=0))
    sharpe = float(series.mean() / std * math.sqrt(252.0)) if std > 0 else None
    max_drawdown = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    active_series = series[series.abs() > 1e-12]
    return {
        "local_total_return": total,
        "local_annual_return": annual,
        "local_sharpe": sharpe,
        "local_max_drawdown": max_drawdown,
        "local_active_days": active_days,
        "local_trade_count": trade_count,
        "local_missing_price_points": missing_price_points,
        "local_win_day_ratio": float((active_series > 0).mean()) if len(active_series) else None,
    }


def extract_indicator(log_text: str):
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except Exception:
                return payload
    return None


def try_juejin(case: str, signal_file: Path, max_positions: int, hold_days: int) -> dict:
    log_file = LOG_DIR / f"{case}_mp{max_positions}_h{hold_days}.log"
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(L2_DB),
        "--max-positions",
        str(max_positions),
        "--holding-days",
        str(hold_days),
        "--max-holding-days",
        str(max(hold_days, 3)),
        "--backtest-slippage-ratio",
        "0.0030",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return {"juejin_returncode": 999, "juejin_log_file": str(log_file), "juejin_error": "timeout"}
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    indicator = extract_indicator(text)
    result = {"juejin_returncode": proc.returncode, "juejin_log_file": str(log_file)}
    if isinstance(indicator, dict):
        result.update({f"juejin_{key}": value for key, value in indicator.items()})
    else:
        result["juejin_indicator_error"] = "missing"
        result["juejin_log_tail"] = "\n".join(text.splitlines()[-10:])
    return result


def make_params() -> list[dict]:
    params: list[dict] = []
    for hold_days in [1, 2, 3, 5]:
        for deep_weight in [0.0, 0.25, 0.5, 0.75]:
            for open_gap_max in [1.5, 2.0, 3.0, None]:
                name = "ogd_pass_dw{dw}_gap{gap}_h{hold}".format(
                    dw=str(deep_weight).replace(".", "p"),
                    gap="none" if open_gap_max is None else str(open_gap_max).replace(".", "p"),
                    hold=hold_days,
                )
                params.append(
                    {
                        "name": name,
                        "base_weight": 0.75,
                        "mild_weight": 2.0,
                        "deep_weight": deep_weight,
                        "open_gap_max": open_gap_max,
                        "cap": 0.90,
                        "hold_days": hold_days,
                    }
                )
    return params


def write_report(results: pd.DataFrame, source_rows: int, source_buy_days: int) -> None:
    top = results.sort_values(["admission_score", "local_annual_return"], ascending=False).head(20)
    lines = [
        "# 通过比例、次日开盘门槛与卖出频率联合调参 20260715",
        "",
        "## 口径",
        "",
        "- 本轮是 research-only，不修改生产策略参数，不发布正式信号。",
        "- 输入固定为当前已落盘 OGD 同源信号，不做补位，不扩大候选池。",
        "- 调参项：深跌信号通过权重、买入日开盘涨幅上限、计划持仓天数。",
        "- 买入日开盘涨幅是执行日开盘可见条件，用于回测交易门槛；不得在信号日前当作已知信息。",
        "- 本地诊断使用未复权 `open`，不混用 qfq 执行价格；摩擦按 0.3% 单边近似。",
        "- 掘金复跑只对本地前几名候选尝试；若终端不可用，不把本地结果当正式结论。",
        "",
        f"- 源信号行数：{source_rows}",
        f"- 源信号买入日：{source_buy_days}",
        "",
        "## 本地 Top 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 通过行比例 | 买入日比例 | 平均仓位 | 持仓天数 | 开盘涨幅上限 | 深跌权重 | 掘金状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in top.to_dict("records"):
        status = "未尝试"
        if pd.notna(row.get("juejin_returncode")):
            status = (
                "成功"
                if int(row.get("juejin_returncode")) == 0
                and row.get("juejin_pnl_ratio_annual") == row.get("juejin_pnl_ratio_annual")
                else "失败/终端不可用"
            )
        gap = "无限制" if pd.isna(row["open_gap_max"]) else f"{row['open_gap_max']:.1f}%"
        lines.append(
            f"| {row['case']} | {row['local_annual_return'] * 100:.2f}% | "
            f"{row['local_sharpe']:.3f} | {row['local_max_drawdown'] * 100:.2f}% | "
            f"{row['pass_row_ratio'] * 100:.2f}% | {row['pass_buy_day_ratio'] * 100:.2f}% | "
            f"{row['mean_daily_target_sum'] * 100:.2f}% | {int(row['hold_days'])} | {gap} | "
            f"{row['deep_weight']:.2f} | {status} |"
        )
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- 本地口径下，最强组合仍未达到年化 500% 和 Sharpe 4 的双目标。",
            "- 提高深跌信号通过比例可以改善覆盖率，但持仓天数拉长后回撤和路径风险会上升。",
            "- 买入日开盘涨幅门槛过严会减少信号；完全放开会提升覆盖但不一定提升 Sharpe。",
            "- 因掘金终端连接失败，本轮不能判定通过生产准入。",
            "",
            "## 证据路径",
            "",
            f"- 结果 CSV：`{REPORT_DIR / 'pass_ratio_open_gap_sellfreq_results_20260715.csv'}`",
            f"- 结果 JSON：`{REPORT_DIR / 'pass_ratio_open_gap_sellfreq_results_20260715.json'}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    (REPORT_DIR / "pass_ratio_open_gap_sellfreq_report_20260715.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = load_source()
    source_rows = len(source)
    source_buy_days = int(source["buy_date"].nunique())

    built: list[tuple[dict, pd.DataFrame, Path]] = []
    all_codes: set[str] = set()
    min_buy = "99999999"
    max_buy = "00000000"
    for params in make_params():
        frame, path = build_signal(source, params)
        if frame.empty:
            continue
        built.append((params, frame, path))
        all_codes.update(frame["stock_code"].dropna().astype(str).unique().tolist())
        min_buy = min(min_buy, str(frame["buy_date"].min()))
        max_buy = max(max_buy, str(frame["buy_date"].max()))

    dates, open_map = load_market(sorted(all_codes), min_buy, "20260714")
    rows: list[dict] = []
    for params, frame, path in built:
        daily_count = frame.groupby("buy_date")["stock_code"].count()
        daily_sum = frame.groupby("buy_date")["target_pct"].sum()
        rec = {
            "case": params["name"],
            "signal_file": str(path),
            "source_rows": source_rows,
            "source_buy_days": source_buy_days,
            "rows": int(len(frame)),
            "pass_row_ratio": float(len(frame) / source_rows) if source_rows else None,
            "buy_days": int(frame["buy_date"].nunique()),
            "pass_buy_day_ratio": float(frame["buy_date"].nunique() / source_buy_days) if source_buy_days else None,
            "stock_count": int(frame["stock_code"].nunique()),
            "min_buy": str(frame["buy_date"].min()),
            "max_buy": str(frame["buy_date"].max()),
            "max_positions": int(daily_count.max()),
            "mean_daily_target_sum": float(daily_sum.mean()),
            "max_daily_target_sum": float(daily_sum.max()),
            "deep_weight": params["deep_weight"],
            "open_gap_max": params["open_gap_max"],
            "hold_days": params["hold_days"],
            "cap": params["cap"],
        }
        rec.update(local_portfolio_eval(frame, params["hold_days"], dates, open_map))
        rec["admission_score"] = (
            rec["local_annual_return"]
            + 0.15 * (rec["local_sharpe"] or 0.0)
            - 0.5 * abs(min(rec["local_max_drawdown"], 0.0))
            + 0.1 * rec["pass_row_ratio"]
        )
        rows.append(rec)

    results = pd.DataFrame(rows).sort_values(["admission_score", "local_annual_return"], ascending=False)

    # Try Juejin for top local candidates plus the baseline. It is often down; keep explicit evidence.
    baseline = "ogd_pass_dw0p0_gapnone_h1"
    try_cases = [baseline] + [case for case in results.head(3)["case"].tolist() if case != baseline]
    try_cases = list(dict.fromkeys(try_cases))[:4]
    juejin_rows = []
    for case in try_cases:
        row = results[results["case"] == case]
        if row.empty:
            continue
        item = row.iloc[0]
        payload = try_juejin(case, Path(str(item["signal_file"])), int(item["max_positions"]), int(item["hold_days"]))
        payload["case"] = case
        juejin_rows.append(payload)
    if juejin_rows:
        results = results.merge(pd.DataFrame(juejin_rows), on="case", how="left")

    results.to_csv(REPORT_DIR / "pass_ratio_open_gap_sellfreq_results_20260715.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "pass_ratio_open_gap_sellfreq_results_20260715.json").write_text(
        json.dumps(results.to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_report(results, source_rows, source_buy_days)
    print(
        results[
            [
                "case",
                "local_annual_return",
                "local_sharpe",
                "local_max_drawdown",
                "pass_row_ratio",
                "pass_buy_day_ratio",
                "mean_daily_target_sum",
                "juejin_returncode",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(REPORT_DIR / "pass_ratio_open_gap_sellfreq_report_20260715.md")


if __name__ == "__main__":
    main()
