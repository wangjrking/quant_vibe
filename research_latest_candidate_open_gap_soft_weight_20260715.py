from __future__ import annotations

import ast
import json
import math
import subprocess
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
BASE_REPORT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_candidate_open_gap_soft_20260715"
SIGNAL_DIR = REPORT_DIR / "signals"
LOG_DIR = REPORT_DIR / "logs"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
)
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"

SOURCES = [
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "ogd_deep8_x1p75_cap90_full.csv",
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "ogd_deep8_x2p0_cap90_full.csv",
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "ogd_deep8_x2p5_cap90_full.csv",
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "ogd_deep8_x3p0_cap90_full.csv",
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "cpo_low60_x1p5_cap90_full.csv",
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "cpo_low60_x2p0_cap90_full.csv",
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "cpo_low60_x2p5_cap90_full.csv",
    BASE_REPORT / "signals" / "latest_candidate_position_lift" / "cpo_low60_x3p0_cap90_full.csv",
]


def load_signal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for column in ["target_pct", "exec_open_gap_pct", "buy_open_gap_pct", "signal_pct_chg_raw", "pred_1d", "pred_10d"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "exec_open_gap_pct" not in frame.columns:
        frame["exec_open_gap_pct"] = pd.to_numeric(frame.get("buy_open_gap_pct"), errors="coerce")
    return frame


def apply_open_gap_soft_weight(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = frame.copy()
    gap = out["exec_open_gap_pct"]
    scale = pd.Series(1.0, index=out.index)
    scale.loc[gap <= -3.0] *= params["deep_low_scale"]
    scale.loc[gap.gt(-3.0) & gap.le(-1.0)] *= params["low_scale"]
    scale.loc[gap.gt(-1.0) & gap.le(0.0)] *= params["slight_low_scale"]
    scale.loc[gap.gt(1.0)] *= params["high_scale"]
    out["target_pct_before_gap_soft"] = out["target_pct"]
    raw = out["target_pct"].fillna(0.0) * scale
    daily = raw.groupby(out["buy_date"]).transform("sum")
    factor = (params["cap"] / daily).clip(upper=1.0).fillna(0.0)
    out["target_pct"] = (raw * factor).clip(lower=0.0)
    out = out[out["target_pct"] > 0].copy()
    out["open_gap_soft_case"] = params["name"]
    out["strategy_variant"] = params["name"]
    out["filter_name"] = params["name"]
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    return out


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


def local_eval(frame: pd.DataFrame, hold_days: int, dates: list[str], open_map: dict[tuple[str, str], float]) -> dict:
    date_index = {date: index for index, date in enumerate(dates)}
    signals = {
        date: [
            {"stock_code": row.stock_code, "target_pct": float(row.target_pct)}
            for row in group.itertuples(index=False)
        ]
        for date, group in frame.groupby("buy_date")
    }
    positions: dict[str, dict] = {}
    returns: list[float] = []
    slippage = 0.0030
    missing = 0
    active_days = 0
    for index, today in enumerate(dates[:-1]):
        next_day = dates[index + 1]
        for code in list(positions):
            if index - positions[code]["entry_index"] >= hold_days:
                del positions[code]
        if today in signals:
            positions = {
                item["stock_code"]: {"target_pct": item["target_pct"], "entry_index": date_index[today]}
                for item in signals[today]
            }
        day_return = 0.0
        if positions:
            active_days += 1
        for code, state in positions.items():
            open_today = open_map.get((code, today))
            open_next = open_map.get((code, next_day))
            if open_today is None or open_next is None:
                missing += 1
                continue
            day_return += state["target_pct"] * (open_next / open_today - 1.0)
        if today in signals:
            day_return -= sum(item["target_pct"] for item in signals[today]) * slippage
        if positions:
            day_return -= sum(state["target_pct"] for state in positions.values()) * (slippage / max(hold_days, 1))
        returns.append(day_return)
    series = pd.Series(returns, index=dates[:-1], dtype=float)
    equity = (1.0 + series).cumprod()
    total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    annual = (float(equity.iloc[-1]) ** (252.0 / len(series)) - 1.0) if len(series) and equity.iloc[-1] > 0 else -1.0
    std = float(series.std(ddof=0))
    sharpe = float(series.mean() / std * math.sqrt(252.0)) if std > 0 else None
    max_drawdown = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    active = series[series.abs() > 1e-12]
    return {
        "local_total_return": total,
        "local_annual_return": annual,
        "local_sharpe": sharpe,
        "local_max_drawdown": max_drawdown,
        "local_active_days": active_days,
        "local_win_day_ratio": float((active > 0).mean()) if len(active) else None,
        "local_missing_price_points": missing,
    }


def extract_indicator(text: str):
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except Exception:
                return payload
    return None


def try_juejin(case: str, signal_file: Path, max_positions: int, hold_days: int) -> dict:
    log = LOG_DIR / f"{case}_mp{max_positions}_h{hold_days}.log"
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log),
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
        return {"juejin_returncode": 999, "juejin_log_file": str(log), "juejin_error": "timeout"}
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    indicator = extract_indicator(text)
    result = {"juejin_returncode": proc.returncode, "juejin_log_file": str(log)}
    if isinstance(indicator, dict):
        result.update({f"juejin_{key}": value for key, value in indicator.items()})
    else:
        result["juejin_indicator_error"] = "missing"
        result["juejin_log_tail"] = "\n".join(text.splitlines()[-10:])
    return result


def param_grid() -> list[dict]:
    rows = []
    for deep_low in [0.8, 1.0, 1.15]:
        for low in [1.0, 1.15, 1.30]:
            for slight_low in [0.95, 1.0, 1.10]:
                for high in [0.25, 0.5, 0.75, 1.0]:
                    for cap in [0.9, 1.0]:
                        name = f"dlg{deep_low}_lg{low}_sl{slight_low}_hi{high}_cap{cap}".replace(".", "p")
                        rows.append(
                            {
                                "name": name,
                                "deep_low_scale": deep_low,
                                "low_scale": low,
                                "slight_low_scale": slight_low,
                                "high_scale": high,
                                "cap": cap,
                            }
                        )
    return rows


def write_report(results: pd.DataFrame) -> None:
    top = results.sort_values(["target_hit", "local_sharpe", "local_annual_return"], ascending=False).head(25)
    lines = [
        "# 最新候选开盘涨幅软权重调参 20260715",
        "",
        "## 口径",
        "",
        "- 输入为覆盖到 20260714 的 `latest_candidate_position_lift` 信号，不使用过期 `max_buy_date=20260528` 的候选。",
        "- 不做日期排除，不扩候选池，只对买入日开盘涨幅分桶做软权重。",
        "- 执行收益本地诊断使用未复权 `open`，摩擦按 0.3% 单边近似。",
        "- 掘金只对本地命中目标或排序靠前候选尝试复跑；终端不可用时只留证据。",
        "",
        "## Top 结果",
        "",
        "| 版本 | 源信号 | 年化 | Sharpe | 最大回撤 | 买入日 | 平均仓位 | 是否命中本地目标 | 掘金状态 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in top.to_dict("records"):
        status = "未尝试"
        if pd.notna(row.get("juejin_returncode")):
            status = (
                "成功"
                if int(row.get("juejin_returncode")) == 0 and pd.notna(row.get("juejin_pnl_ratio_annual"))
                else "失败/终端不可用"
            )
        lines.append(
            f"| {row['case']} | {row['source_name']} | {row['local_annual_return'] * 100:.2f}% | "
            f"{row['local_sharpe']:.3f} | {abs(row['local_max_drawdown']) * 100:.2f}% | "
            f"{int(row['buy_days'])} | {row['mean_daily_target_sum'] * 100:.2f}% | "
            f"{'是' if row['target_hit'] else '否'} | {status} |"
        )
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- 本地口径用于筛选方向，正式准入仍必须等掘金复跑。",
            "- 若本地命中但掘金不可用，只能列为待复跑候选。",
            "",
            "## 证据路径",
            "",
            f"- 结果 CSV：`{REPORT_DIR / 'latest_candidate_open_gap_soft_results_20260715.csv'}`",
            f"- 结果 JSON：`{REPORT_DIR / 'latest_candidate_open_gap_soft_results_20260715.json'}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    (REPORT_DIR / "latest_candidate_open_gap_soft_report_20260715.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    source_frames = []
    all_codes: set[str] = set()
    min_buy = "99999999"
    max_buy = "00000000"
    for path in SOURCES:
        if not path.exists():
            continue
        frame = load_signal(path)
        source_frames.append((path, frame))
        all_codes.update(frame["stock_code"].astype(str).unique().tolist())
        min_buy = min(min_buy, str(frame["buy_date"].min()))
        max_buy = max(max_buy, str(frame["buy_date"].max()))
    dates, open_map = load_market(sorted(all_codes), min_buy, "20260714")

    rows = []
    built: dict[str, tuple[Path, int, int]] = {}
    for path, frame in source_frames:
        source_name = path.stem
        for params in param_grid():
            out = apply_open_gap_soft_weight(frame, params)
            if out.empty:
                continue
            case = f"{source_name}_{params['name']}"
            out["strategy_variant"] = case
            out["filter_name"] = case
            out_path = SIGNAL_DIR / f"{case}.csv"
            out.to_csv(out_path, index=False, encoding="utf-8-sig")
            max_positions = int(out.groupby("buy_date")["stock_code"].count().max())
            daily_sum = out.groupby("buy_date")["target_pct"].sum()
            rec = {
                "case": case,
                "source_name": source_name,
                "source_file": str(path),
                "signal_file": str(out_path),
                "rows": int(len(out)),
                "buy_days": int(out["buy_date"].nunique()),
                "stock_count": int(out["stock_code"].nunique()),
                "min_buy": str(out["buy_date"].min()),
                "max_buy": str(out["buy_date"].max()),
                "max_positions": max_positions,
                "mean_daily_target_sum": float(daily_sum.mean()),
                "max_daily_target_sum": float(daily_sum.max()),
                "hold_days": 1,
                **params,
            }
            rec.update(local_eval(out, 1, dates, open_map))
            rec["target_hit"] = (
                rec["local_annual_return"] >= 5.0
                and (rec["local_sharpe"] or 0.0) >= 4.0
                and abs(rec["local_max_drawdown"]) <= 0.40
                and rec["max_buy"] >= "20260714"
            )
            rows.append(rec)
            built[case] = (out_path, max_positions, 1)

    results = pd.DataFrame(rows).sort_values(["target_hit", "local_sharpe", "local_annual_return"], ascending=False)
    try_cases = results[results["target_hit"]].head(2)["case"].tolist()
    if len(try_cases) < 2:
        try_cases += [case for case in results.head(2)["case"].tolist() if case not in try_cases]
    juejin_rows = []
    for case in try_cases[:3]:
        signal_file, max_positions, hold_days = built[case]
        payload = try_juejin(case, signal_file, max_positions, hold_days)
        payload["case"] = case
        juejin_rows.append(payload)
    if juejin_rows:
        results = results.merge(pd.DataFrame(juejin_rows), on="case", how="left")

    results.to_csv(REPORT_DIR / "latest_candidate_open_gap_soft_results_20260715.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "latest_candidate_open_gap_soft_results_20260715.json").write_text(
        json.dumps(results.to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_report(results)
    print(
        results[
            [
                "case",
                "local_annual_return",
                "local_sharpe",
                "local_max_drawdown",
                "buy_days",
                "mean_daily_target_sum",
                "target_hit",
                "juejin_returncode",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(REPORT_DIR / "latest_candidate_open_gap_soft_report_20260715.md")


if __name__ == "__main__":
    main()
