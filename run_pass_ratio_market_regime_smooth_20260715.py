from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_full_refill_scale_smooth_20260715.py"

spec = importlib.util.spec_from_file_location("smooth_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SOURCE = REPORT_DIR / "signals" / "full_refill_scale_smooth" / "frs_scale065_cap070_full.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "market_regime_smooth"
LOG_DIR = REPORT_DIR / "logs" / "market_regime_smooth_20260715"
OUT_CSV = REPORT_DIR / "market_regime_smooth_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "market_regime_smooth_juejin_20260715.json"
OUT_MD = REPORT_DIR / "market_regime_smooth_review_20260715.md"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {"case": "mrs_r5neg130_flat100_pos070_cap090", "neg": 1.30, "flat": 1.00, "pos": 0.70, "strong_neg": 1.45, "cap": 0.90},
    {"case": "mrs_r5neg140_flat100_pos060_cap095", "neg": 1.40, "flat": 1.00, "pos": 0.60, "strong_neg": 1.60, "cap": 0.95},
    {"case": "mrs_r5neg150_flat095_pos050_cap100", "neg": 1.50, "flat": 0.95, "pos": 0.50, "strong_neg": 1.75, "cap": 1.00},
    {"case": "mrs_r5neg160_flat090_pos040_cap100", "neg": 1.60, "flat": 0.90, "pos": 0.40, "strong_neg": 1.90, "cap": 1.00},
    {"case": "mrs_r5neg170_flat085_pos030_cap100", "neg": 1.70, "flat": 0.85, "pos": 0.30, "strong_neg": 2.00, "cap": 1.00},
    {"case": "mrs_r3neg140_flat100_pos060_cap095", "neg": 1.40, "flat": 1.00, "pos": 0.60, "strong_neg": 1.65, "cap": 0.95, "use": "idx_ret3"},
    {"case": "mrs_r3neg155_flat095_pos050_cap100", "neg": 1.55, "flat": 0.95, "pos": 0.50, "strong_neg": 1.85, "cap": 1.00, "use": "idx_ret3"},
    {"case": "mrs_r3neg170_flat090_pos040_cap100", "neg": 1.70, "flat": 0.90, "pos": 0.40, "strong_neg": 2.05, "cap": 1.00, "use": "idx_ret3"},
    {"case": "mrs_combo150_095_050_cap100", "neg": 1.50, "flat": 0.95, "pos": 0.50, "strong_neg": 1.75, "cap": 1.00, "combo": True},
    {"case": "mrs_combo165_090_040_cap100", "neg": 1.65, "flat": 0.90, "pos": 0.40, "strong_neg": 1.95, "cap": 1.00, "combo": True},
]


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def num(value: object, default: float = 0.0) -> float:
    if value is None or value == "" or pd.isna(value):
        return default
    return float(value)


def integer(value: object, default: int = 0) -> int:
    if value is None or value == "" or pd.isna(value):
        return default
    return int(float(value))


def load_index_state() -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        idx = con.execute(
            """
            select distinct trade_date, index_2000_close
            from STOCK_DAILY_DATA
            where trade_date between '20220501' and '20260714'
            order by trade_date
            """
        ).fetchdf()
    finally:
        con.close()
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx = idx.sort_values("trade_date")
    idx["idx_ret3"] = idx["index_2000_close"].pct_change(3)
    idx["idx_ret5"] = idx["index_2000_close"].pct_change(5)
    return idx[["trade_date", "idx_ret3", "idx_ret5"]]


def build_case(source: pd.DataFrame, idx: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = source.merge(idx, left_on="signal_date", right_on="trade_date", how="left").copy()
    ret = pd.to_numeric(df["idx_ret5"], errors="coerce")
    if case.get("use") == "idx_ret3":
        ret = pd.to_numeric(df["idx_ret3"], errors="coerce")
    elif case.get("combo"):
        ret = pd.concat(
            [pd.to_numeric(df["idx_ret3"], errors="coerce"), pd.to_numeric(df["idx_ret5"], errors="coerce")],
            axis=1,
        ).min(axis=1)

    scale = pd.Series(float(case["flat"]), index=df.index)
    scale.loc[ret > 0.0] = float(case["pos"])
    scale.loc[(ret <= 0.0) & (ret > -0.02)] = float(case["flat"])
    scale.loc[(ret <= -0.02) & (ret > -0.05)] = float(case["neg"])
    scale.loc[ret <= -0.05] = float(case["strong_neg"])

    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    df["target_pct"] = (target * scale).clip(lower=0.0, upper=float(case["cap"]))
    df = df[df["target_pct"] > 0].copy()
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = df["target_pct"] * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["market_regime_smooth_case"] = case["case"]
    df["rank"] = df.groupby("buy_date")["target_pct"].rank(method="first", ascending=False).astype(int)
    return df.sort_values(["buy_date", "rank"]).copy()


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    idx = load_index_state()
    rows = []
    for case in CASES:
        signal = build_case(source, idx, case)
        out_path = SIGNAL_DIR / f"{case['case']}_full.csv"
        signal.to_csv(out_path, index=False, encoding="utf-8-sig")
        max_positions = int(max(signal.groupby("buy_date")["stock_code"].count().max(), 1)) if len(signal) else 1
        log_file = LOG_DIR / f"{case['case']}_full_mp{max_positions}.log"
        result = base.run_juejin(out_path, log_file, max_positions)
        daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
        result.update(
            {
                **case,
                "signal_file": str(out_path),
                "rows": int(len(signal)),
                "buy_days": int(signal["buy_date"].nunique()) if len(signal) else 0,
                "stock_count": int(signal["stock_code"].nunique()) if len(signal) else 0,
                "max_positions": max_positions,
                "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                "max_buy_date": str(signal["buy_date"].max()) if len(signal) else None,
            }
        )
        rows.append(result)
        print(json.dumps({"case": case["case"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "buy_days": result["buy_days"]}, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 市场状态分层仓位验证 20260715",
        "",
        "## 口径",
        "",
        "- 输入：`frs_scale065_cap070_full` 高覆盖信号，覆盖到 20260714。",
        "- 用信号日已知的 `index_2000_close` 近 3/5 日涨跌幅调仓。",
        "- 市场回调阶段加仓，市场上涨阶段降仓；不新增候补票、不使用未来收益。",
        "- 本轮为 research-only，指标以掘金日志为准。",
        "",
        "## full 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 买入日 | 开仓 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | {num(row.get('sharp_ratio')):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {integer(row.get('buy_days'))} | {integer(row.get('open_count'))} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy_date')} |"
        )
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
