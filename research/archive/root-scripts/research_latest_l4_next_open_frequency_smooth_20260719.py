from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
PRODUCTION_DIR = MAIN / "strategy_library" / "production" / "prod_high_return_frs_scale090_cap090_v20260716"
SOURCE_SIGNAL = PRODUCTION_DIR / "signals" / "full_history_high_return_frs_scale090_cap090.csv"
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
    / "code_snapshots"
    / "replacement_exit"
)
RUNNER = MAIN / "run_juejin_signal_backtest.py"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_next_open_frequency_smooth_20260719"
SIGNAL_DIR = REPORT_DIR / "signals"
LOG_DIR = REPORT_DIR / "logs"
LOCAL_SUMMARY = REPORT_DIR / "buy_weight_local_screen.csv"
STABILITY_SUMMARY = REPORT_DIR / "buy_weight_stability_screen.csv"
JUEJIN_SUMMARY = REPORT_DIR / "juejin_full_results.csv"
RESULT_JSON = REPORT_DIR / "juejin_full_results.json"
REPORT_MD = REPORT_DIR / "研究报告.md"

BACKTEST_SLIPPAGE_RATIO = 0.003


BUY_PROFILES = [
    {"name": "base_cap90", "cap": 0.90, "pos_gap_penalty": 0.00, "deep_gap_boost": 0.00, "deep_drop_boost": 0.00, "weak_signal_penalty": 0.00, "floor": 1.00, "ceiling": 1.00},
    {"name": "evidence_mild_cap90", "cap": 0.90, "pos_gap_penalty": 0.04, "deep_gap_boost": 0.02, "deep_drop_boost": 0.025, "weak_signal_penalty": 0.08, "floor": 0.65, "ceiling": 1.20},
    {"name": "evidence_mild_cap85", "cap": 0.85, "pos_gap_penalty": 0.04, "deep_gap_boost": 0.02, "deep_drop_boost": 0.025, "weak_signal_penalty": 0.08, "floor": 0.65, "ceiling": 1.20},
    {"name": "evidence_balanced_cap90", "cap": 0.90, "pos_gap_penalty": 0.07, "deep_gap_boost": 0.03, "deep_drop_boost": 0.040, "weak_signal_penalty": 0.14, "floor": 0.45, "ceiling": 1.35},
    {"name": "evidence_balanced_cap85", "cap": 0.85, "pos_gap_penalty": 0.07, "deep_gap_boost": 0.03, "deep_drop_boost": 0.040, "weak_signal_penalty": 0.14, "floor": 0.45, "ceiling": 1.35},
    {"name": "evidence_strict_cap80", "cap": 0.80, "pos_gap_penalty": 0.10, "deep_gap_boost": 0.04, "deep_drop_boost": 0.055, "weak_signal_penalty": 0.20, "floor": 0.30, "ceiling": 1.45},
    {"name": "grid_center_cap85", "cap": 0.85, "pos_gap_penalty": 0.05, "deep_gap_boost": 0.00, "deep_drop_boost": 0.100, "weak_signal_penalty": 0.30, "floor": 0.20, "ceiling": 1.60},
    {"name": "grid_left_cap85", "cap": 0.85, "pos_gap_penalty": 0.05, "deep_gap_boost": 0.00, "deep_drop_boost": 0.075, "weak_signal_penalty": 0.25, "floor": 0.20, "ceiling": 1.60},
    {"name": "grid_right_cap85", "cap": 0.85, "pos_gap_penalty": 0.05, "deep_gap_boost": 0.00, "deep_drop_boost": 0.125, "weak_signal_penalty": 0.35, "floor": 0.20, "ceiling": 1.60},
    {"name": "grid_center_cap90", "cap": 0.90, "pos_gap_penalty": 0.05, "deep_gap_boost": 0.00, "deep_drop_boost": 0.100, "weak_signal_penalty": 0.30, "floor": 0.20, "ceiling": 1.60},
    {"name": "robust_left_cap90", "cap": 0.90, "pos_gap_penalty": 0.05, "deep_gap_boost": 0.00, "deep_drop_boost": 0.150, "weak_signal_penalty": 0.45, "floor": 0.10, "ceiling": 1.80},
    {"name": "robust_center_cap90", "cap": 0.90, "pos_gap_penalty": 0.05, "deep_gap_boost": 0.00, "deep_drop_boost": 0.200, "weak_signal_penalty": 0.50, "floor": 0.10, "ceiling": 1.90},
    {"name": "robust_right_cap90", "cap": 0.90, "pos_gap_penalty": 0.05, "deep_gap_boost": 0.00, "deep_drop_boost": 0.225, "weak_signal_penalty": 0.55, "floor": 0.05, "ceiling": 1.90},
]

SELL_PROFILES = [
    {"name": "baseline_h1_mh3_e098_c102", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 0.98, "continue_ratio": 1.02, "replace_mode": False, "replace_ratio": 9.99, "replace_edge": 9.99},
    {"name": "replace_min1_mh2_r095_e000", "holding_days": 1, "max_holding_days": 2, "exit_ratio": 0.0, "continue_ratio": 0.0, "replace_mode": True, "replace_ratio": 0.95, "replace_edge": 0.0},
    {"name": "replace_min1_mh2_r100_e000", "holding_days": 1, "max_holding_days": 2, "exit_ratio": 0.0, "continue_ratio": 0.0, "replace_mode": True, "replace_ratio": 1.00, "replace_edge": 0.0},
    {"name": "replace_min1_mh2_r100_e005", "holding_days": 1, "max_holding_days": 2, "exit_ratio": 0.0, "continue_ratio": 0.0, "replace_mode": True, "replace_ratio": 1.00, "replace_edge": 0.005},
    {"name": "replace_min1_mh2_r102_e000", "holding_days": 1, "max_holding_days": 2, "exit_ratio": 0.0, "continue_ratio": 0.0, "replace_mode": True, "replace_ratio": 1.02, "replace_edge": 0.0},
    {"name": "replace_min1_mh3_r100_e000", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 0.0, "continue_ratio": 0.0, "replace_mode": True, "replace_ratio": 1.00, "replace_edge": 0.0},
]


def _number(value: object, default: float = 0.0) -> float:
    if value is None or value == "" or pd.isna(value):
        return default
    return float(value)


def load_source() -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(
        SOURCE_SIGNAL,
        encoding="utf-8-sig",
        dtype={"signal_date": str, "buy_date": str, "stock_code": str},
    )
    source["signal_date"] = source["signal_date"].str.replace(".0", "", regex=False).str.zfill(8)
    source["buy_date"] = source["buy_date"].str.replace(".0", "", regex=False).str.zfill(8)
    source = source[~source["stock_code"].str.endswith(".BJ", na=False)].copy()
    if source.empty:
        raise RuntimeError("No mature production signals available")

    con = duckdb.connect(str(MARKET_DB), read_only=True)
    calendar = con.execute("select distinct cast(trade_date as varchar) as trade_date from STOCK_DAILY_DATA order by 1").fetchdf()
    market_max = str(calendar["trade_date"].max())
    dates = calendar["trade_date"].astype(str).tolist()
    mature_cutoff = dates[-5] if len(dates) >= 5 else dates[0]
    returns = con.execute(
        """
        with cal as (
            select trade_date, lead(trade_date) over (order by trade_date) as next_trade_date
            from (select distinct trade_date from STOCK_DAILY_DATA)
        )
        select
            cast(b.trade_date as varchar) as buy_date,
            b.stock_code,
            cast(cal.next_trade_date as varchar) as next_trade_date,
            b.open as buy_open_raw,
            b.pre_close as buy_pre_close_raw,
            b.name as buy_name,
            b.ST_TYPE as buy_st_type,
            b.ST_TYPE_name as buy_st_type_name,
            n.open as next_open_raw,
            n.open / nullif(b.open, 0) - 1.0 as next_open_return_raw
        from STOCK_DAILY_DATA b
        join cal on cal.trade_date = b.trade_date
        join STOCK_DAILY_DATA n
          on n.trade_date = cal.next_trade_date and n.stock_code = b.stock_code
        where cal.next_trade_date is not null
        """
    ).fetchdf()
    con.close()
    returns["buy_date"] = returns["buy_date"].astype(str)
    returns["stock_code"] = returns["stock_code"].astype(str)
    source = source[source["buy_date"] <= mature_cutoff].copy()
    source = source.merge(returns, on=["buy_date", "stock_code"], how="inner")
    board_limit = np.where(
        source["stock_code"].str.startswith(("300", "301", "688"), na=False),
        0.20,
        0.10,
    )
    st_type = source["buy_st_type"].fillna("").astype(str).str.strip().str.lower()
    st_name = source["buy_st_type_name"].fillna("").astype(str).str.strip().str.lower()
    buy_name = source["buy_name"].fillna(source.get("name", "")).astype(str).str.strip()
    normal_st_type = st_type.isin({"", "0", "0.0", "nan", "none"})
    normal_st_name = st_name.isin({"", "0", "0.0", "nan", "none", "正常", "无"})
    no_name_risk = ~buy_name.str.match(r"^(?:\*?ST|退市)", case=False, na=False)
    not_open_limit_up = source["buy_open_raw"] < source["buy_pre_close_raw"] * (1.0 + board_limit) * 0.995
    source = source[normal_st_type & normal_st_name & no_name_risk & not_open_limit_up].copy()
    for column in ["target_pct", "exec_open_gap_pct", "buy_open_gap_raw_pct", "signal_pct_chg_raw"]:
        source[column] = pd.to_numeric(source.get(column), errors="coerce")
    source["open_gap"] = source["exec_open_gap_pct"].fillna(source["buy_open_gap_raw_pct"]).fillna(0.0)
    source["signal_drop"] = source["signal_pct_chg_raw"].fillna(0.0)
    source["target_pct"] = source["target_pct"].fillna(0.0).clip(lower=0.0)
    audit = {
        "source_rows": int(len(source)),
        "signal_days": int(source["signal_date"].nunique()),
        "buy_days": int(source["buy_date"].nunique()),
        "min_signal_date": str(source["signal_date"].min()),
        "max_signal_date": str(source["signal_date"].max()),
        "min_buy_date": str(source["buy_date"].min()),
        "max_buy_date": str(source["buy_date"].max()),
        "market_max_date": market_max,
        "mature_cutoff": mature_cutoff,
        "bj_rows": int(source["stock_code"].str.endswith(".BJ", na=False).sum()),
        "duplicate_keys": int(source.duplicated(["buy_date", "stock_code"]).sum()),
        "hard_gate_replayed_from_current_l2": True,
    }
    return source, audit


def apply_buy_profile(source: pd.DataFrame, profile: dict) -> pd.DataFrame:
    out = source.copy()
    positive_gap = np.maximum(out["open_gap"] - 0.5, 0.0)
    deep_gap = np.maximum(-4.0 - out["open_gap"], 0.0)
    deep_drop = np.maximum(-4.0 - out["signal_drop"], 0.0)
    weak_signal = np.maximum(out["signal_drop"] + 2.0, 0.0)
    scale = (
        1.0
        - float(profile["pos_gap_penalty"]) * positive_gap
        + float(profile["deep_gap_boost"]) * deep_gap
        + float(profile["deep_drop_boost"]) * deep_drop
        - float(profile["weak_signal_penalty"]) * weak_signal
    )
    out["next_open_weight_scale"] = np.clip(scale, float(profile["floor"]), float(profile["ceiling"]))
    out["target_pct"] = out["target_pct"] * out["next_open_weight_scale"]
    daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = np.minimum(1.0, float(profile["cap"]) / daily_sum.replace(0.0, np.nan)).fillna(1.0)
    out["target_pct"] = out["target_pct"] * cap_scale
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = profile["name"]
    out["filter_name"] = profile["name"]
    return out


def _period_stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, mdd


def local_stats(frame: pd.DataFrame) -> dict:
    work = frame.copy()
    work["weighted_return"] = work["target_pct"] * (
        work["next_open_return_raw"] - 2.0 * BACKTEST_SLIPPAGE_RATIO
    )
    daily = work.groupby("buy_date")["weighted_return"].sum().sort_index()
    full_dates = pd.date_range(pd.to_datetime(daily.index.min()), pd.to_datetime(daily.index.max()), freq="B")
    daily = daily.reindex(full_dates.strftime("%Y%m%d"), fill_value=0.0)
    annual, sharpe, mdd = _period_stats(daily)
    recent = daily.tail(60)
    _, recent_sharpe, _ = _period_stats(recent)

    dated = daily.copy()
    dated.index = pd.to_datetime(dated.index)
    yearly = [_period_stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 40]
    rolling = [_period_stats(daily.iloc[start : start + 60]) for start in range(0, max(len(daily) - 59, 1), 20)]
    positive = daily[daily > 0].sort_values(ascending=False)
    positive_sum = float(positive.sum())
    top10_concentration = float(positive.head(10).sum() / positive_sum) if positive_sum > 0 else 1.0
    return {
        "local_annual": annual,
        "local_sharpe": sharpe,
        "local_mdd": mdd,
        "local_recent60_sharpe": recent_sharpe,
        "local_worst_year_annual": min((item[0] for item in yearly), default=0.0),
        "local_worst_year_sharpe": min((item[1] for item in yearly), default=0.0),
        "local_rolling60_min_annual": min((item[0] for item in rolling), default=0.0),
        "local_rolling60_min_sharpe": min((item[1] for item in rolling), default=0.0),
        "local_rolling60_median_sharpe": float(np.median([item[1] for item in rolling])) if rolling else 0.0,
        "local_top10_positive_concentration": top10_concentration,
        "mean_daily_target": float(work.groupby("buy_date")["target_pct"].sum().mean()),
        "min_daily_target": float(work.groupby("buy_date")["target_pct"].sum().min()),
        "max_daily_target": float(work.groupby("buy_date")["target_pct"].sum().max()),
    }


def parse_indicator(text: str) -> dict:
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(text.splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result = {}
            for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "risk_ratio", "win_ratio", "calmar_ratio"]:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            for key in ["open_count", "close_count", "win_count", "lose_count"]:
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    result[key] = int(match.group(1))
            return result
    return {}


def run_juejin(signal_file: Path, buy_profile: dict, sell_profile: dict, audit: dict) -> dict:
    case = f"{buy_profile['name']}__{sell_profile['name']}"
    log_file = LOG_DIR / f"{case}.log"
    effective_signal_file = SIGNAL_DIR / f"{case}.csv"
    effective_signal = pd.read_csv(signal_file, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    effective_signal["holding_days"] = int(sell_profile["holding_days"])
    effective_signal["max_holding_days"] = int(sell_profile["max_holding_days"])
    effective_signal["score_exit_entry_ratio"] = float(sell_profile["exit_ratio"])
    effective_signal["score_continue_entry_ratio"] = float(sell_profile["continue_ratio"])
    effective_signal["min_holding_days_before_score_exit"] = 99 if sell_profile["replace_mode"] else 1
    effective_signal["effective_sell_profile"] = sell_profile["name"]
    effective_signal.to_csv(effective_signal_file, index=False, encoding="utf-8-sig")
    command = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(effective_signal_file),
        "--log-file", str(log_file),
        "--max-positions", "2",
        "--holding-days", str(sell_profile["holding_days"]),
        "--max-holding-days", str(sell_profile["max_holding_days"]),
        "--score-exit-entry-ratio", str(sell_profile["exit_ratio"]),
        "--score-continue-entry-ratio", str(sell_profile["continue_ratio"]),
        "--min-holding-days-before-score-exit", "1",
        "--score-db", str(SCORE_DB),
        "--score-table", SCORE_TABLE,
        "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none",
        "--backtest-slippage-ratio", str(BACKTEST_SLIPPAGE_RATIO),
        "--backtest-start", "2022-08-12 09:00:00",
        "--backtest-end", f"{audit['market_max_date'][:4]}-{audit['market_max_date'][4:6]}-{audit['market_max_date'][6:]} 15:30:00",
    ]
    env = os.environ.copy()
    env.update(
        {
            "GM_REPLACE_EXIT_MODE": "1" if sell_profile["replace_mode"] else "0",
            "GM_REPLACE_EXIT_MIN_HOLDING_DAYS": "1",
            "GM_REPLACE_EXIT_SCORE_RATIO": str(sell_profile["replace_ratio"]),
            "GM_REPLACE_EXIT_MIN_EDGE": str(sell_profile["replace_edge"]),
            "GM_REPLACE_EXIT_USE_ENTRY_FALLBACK": "0",
        }
    )
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    log_text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    indicator = parse_indicator(log_text)
    return {
        "case": case,
        "buy_profile": buy_profile["name"],
        "sell_profile": sell_profile["name"],
        "returncode": int(proc.returncode),
        "signal_file": str(effective_signal_file),
        "log_file": str(log_file),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "pnl_ratio": indicator.get("pnl_ratio"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "stderr_tail": proc.stderr[-500:] if proc.returncode else "",
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source, audit = load_source()

    local_rows = []
    built = {}
    for profile in BUY_PROFILES:
        frame = apply_buy_profile(source, profile)
        signal_file = SIGNAL_DIR / f"{profile['name']}.csv"
        frame.to_csv(signal_file, index=False, encoding="utf-8-sig")
        stats = local_stats(frame)
        local_rows.append({**profile, **stats, "signal_file": str(signal_file)})
        built[profile["name"]] = signal_file
    local = pd.DataFrame(local_rows).sort_values(["local_sharpe", "local_annual"], ascending=False)
    local.to_csv(LOCAL_SUMMARY, index=False, encoding="utf-8-sig")
    local.sort_values(
        ["local_rolling60_median_sharpe", "local_worst_year_sharpe", "local_sharpe"],
        ascending=False,
    ).to_csv(STABILITY_SUMMARY, index=False, encoding="utf-8-sig")

    selected_names = list(dict.fromkeys(["base_cap90", *local.head(3)["name"].tolist()]))
    selected_profiles = [profile for profile in BUY_PROFILES if profile["name"] in selected_names]
    results = []
    terminal_unavailable = False
    skip_juejin = str(os.environ.get("STRATEGY_SKIP_JUEJIN", "0")).strip().lower() in {"1", "true", "yes"}
    if not skip_juejin:
        for buy_profile in selected_profiles:
            for sell_profile in SELL_PROFILES:
                result = run_juejin(built[buy_profile["name"]], buy_profile, sell_profile, audit)
                results.append(result)
                print(json.dumps({key: result.get(key) for key in ["case", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
                if result["returncode"] != 0:
                    log_text = Path(result["log_file"]).read_text(encoding="utf-8", errors="ignore") if Path(result["log_file"]).exists() else ""
                    if "1001" in log_text or "终端服务" in log_text:
                        terminal_unavailable = True
                        break
            if terminal_unavailable:
                break

    frame = pd.DataFrame(results)
    if frame.empty:
        frame = pd.DataFrame(columns=["case", "buy_profile", "sell_profile", "returncode", "annual_return", "sharpe", "max_drawdown", "target_hit"])
    else:
        frame["target_hit"] = (
            (pd.to_numeric(frame["annual_return"], errors="coerce") >= 5.0)
            & (pd.to_numeric(frame["sharpe"], errors="coerce") >= 4.0)
            & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.40)
        )
        frame = frame.sort_values(["target_hit", "sharpe", "annual_return"], ascending=[False, False, False])
    frame.to_csv(JUEJIN_SUMMARY, index=False, encoding="utf-8-sig")
    RESULT_JSON.write_text(
        json.dumps(
            {
                "status": "research_only",
                "input_contract": {
                    "source_signal": str(SOURCE_SIGNAL),
                    "score_db": str(SCORE_DB),
                    "score_table": SCORE_TABLE,
                    "market_db": str(MARKET_DB),
                    "backtest_adjust": "none",
                    "base_slippage_ratio": BACKTEST_SLIPPAGE_RATIO,
                    "adaptive_slippage_in_code_snapshot": True,
                },
                "audit": audit,
                "local_screen": local.to_dict("records"),
                "juejin_results": frame.to_dict("records"),
                "target_hit_count": int(frame["target_hit"].sum()),
                "terminal_unavailable": terminal_unavailable,
                "juejin_skipped": skip_juejin,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 最新 L4 次日开盘与买卖频率平滑研究",
        "",
        "## 研究口径",
        "",
        f"- 使用当前生产策略完整信号作为基线，并使用 active formal 10D DuckDB 日评分做卖出判断。",
        f"- 回测执行价格为不复权开盘价，`backtest_adjust=none`；基础滑点为 `{BACKTEST_SLIPPAGE_RATIO:.4f}`，代码快照同时启用自适应流动性滑点。",
        "- 次日开盘规则只做连续仓位缩放，不删除信号，不使用日期、月份、行业或个股特例。",
        "- 本报告为 research-only，不修改生产策略。",
        "",
        "## 掘金结果",
        "",
        "| 候选 | 年化 | Sharpe | 最大回撤 | 胜率 | 开仓 | 平仓 | 达标 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.head(20).to_dict("records"):
        annual = _number(row.get("annual_return")) * 100
        mdd = _number(row.get("max_drawdown")) * 100
        win = _number(row.get("win_ratio")) * 100
        lines.append(
            f"| `{row['case']}` | {annual:.2f}% | {_number(row.get('sharpe')):.3f} | {mdd:.2f}% | "
            f"{win:.2f}% | {int(_number(row.get('open_count')))} | {int(_number(row.get('close_count')))} | "
            f"{'是' if row.get('target_hit') else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 本地粗筛：`{LOCAL_SUMMARY}`",
            f"- 本地稳健性筛选：`{STABILITY_SUMMARY}`",
            f"- 掘金结果：`{JUEJIN_SUMMARY}`",
            f"- 结构化结果：`{RESULT_JSON}`",
            f"- 掘金日志：`{LOG_DIR}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT_MD), "target_hit_count": int(frame["target_hit"].sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
