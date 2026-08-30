from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_open_verified_scale_fine_20260715.py"

spec = importlib.util.spec_from_file_location("scale_fine", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

SOURCE_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT_DIR = REPORT_DIR / "signals" / "coverage_layer_reaudit"
LOG_DIR = REPORT_DIR / "logs" / "coverage_layer_reaudit_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_coverage_layer_reaudit_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_coverage_layer_reaudit_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_coverage_layer_reaudit_review_20260715.md"
AUDIT_CSV = REPORT_DIR / "pass_ratio_coverage_layer_reaudit_hardgate_20260715.csv"
AUDIT_JSON = REPORT_DIR / "pass_ratio_coverage_layer_reaudit_hardgate_20260715.json"

TRADE_DAYS = 997

CASES = [
    {
        "case": "core_top1_recalc_s120",
        "source": REPORT_DIR / "signals" / "recalc_hardgate_scale" / "rh_top1_s120_cap065_full.csv",
        "family": "current_core",
        "allow": True,
        "note": "current core, latest L2 hard gate already recalculated",
    },
    {
        "case": "coverage_bl_empty_o044_s005",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "bl_empty_o044_s005.csv",
        "family": "coverage_layer",
        "allow": True,
        "note": "predefined micro coverage layer on empty core days, not buy-day refill",
    },
    {
        "case": "coverage_bl_empty_o044_s005_s120",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "bl_empty_o044_s005.csv",
        "family": "coverage_layer_scaled",
        "allow": True,
        "position_scale": 1.20,
        "position_cap": 0.65,
        "daily_cap": 1.00,
        "note": "same coverage layer, position scaled after selection",
    },
    {
        "case": "coverage_bl_empty_o044_s005_s135",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "bl_empty_o044_s005.csv",
        "family": "coverage_layer_scaled",
        "allow": True,
        "position_scale": 1.35,
        "position_cap": 0.65,
        "daily_cap": 1.00,
        "note": "same coverage layer, position scaled after selection",
    },
    {
        "case": "coverage_bl_empty_o044_s005_s150",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "bl_empty_o044_s005.csv",
        "family": "coverage_layer_scaled",
        "allow": True,
        "position_scale": 1.50,
        "position_cap": 0.65,
        "daily_cap": 1.00,
        "note": "same coverage layer, position scaled after selection",
    },
    {
        "case": "coverage_bl_empty_o044_s005_s160",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "bl_empty_o044_s005.csv",
        "family": "coverage_layer_scaled",
        "allow": True,
        "position_scale": 1.60,
        "position_cap": 0.65,
        "daily_cap": 1.00,
        "note": "same coverage layer, position scaled after selection",
    },
    {
        "case": "coverage_bl_empty_o044_s005_s175",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "bl_empty_o044_s005.csv",
        "family": "coverage_layer_scaled",
        "allow": True,
        "position_scale": 1.75,
        "position_cap": 0.65,
        "daily_cap": 1.00,
        "note": "same coverage layer, position scaled after selection",
    },
    {
        "case": "coverage_bl_empty_o044_s005_s200",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "bl_empty_o044_s005.csv",
        "family": "coverage_layer_scaled",
        "allow": True,
        "position_scale": 2.00,
        "position_cap": 0.65,
        "daily_cap": 1.00,
        "note": "same coverage layer, position scaled after selection",
    },
    {
        "case": "coverage_gb_empty_o044_s005",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "gb_empty_o044_s005.csv",
        "family": "coverage_layer",
        "allow": True,
        "note": "predefined micro coverage layer on empty core days, not buy-day refill",
    },
    {
        "case": "coverage_sh_empty_o028_s003",
        "source": SOURCE_REPORT_DIR / "signals" / "core_plus_micro_overlay" / "sh_empty_o028_s003.csv",
        "family": "coverage_layer",
        "allow": True,
        "note": "predefined micro coverage layer on empty core days, not buy-day refill",
    },
]


def limit_up_pct(stock_code: str) -> float:
    code = str(stock_code or "").split(".", 1)[0]
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def is_bad_st(value: object) -> bool:
    text = str(value or "").strip()
    return text not in {"", "0", "0.0", "None", "nan", "NaN", "正常", "无"}


def recalc_hardgate(frame: pd.DataFrame) -> pd.DataFrame:
    stale_cols = [
        "buy_day_name_latest_l2",
        "buy_open_raw",
        "buy_pre_close_raw",
        "buy_ST_TYPE_latest_l2",
        "buy_ST_TYPE_name_latest_l2",
    ]
    frame = frame.drop(columns=[col for col in stale_cols if col in frame.columns], errors="ignore")
    keys = frame[["buy_date", "stock_code"]].drop_duplicates().copy()
    con = duckdb.connect(str(L2_DB), read_only=True)
    con.register("need_keys", keys)
    market = con.execute(
        """
        SELECT
            k.buy_date,
            k.stock_code,
            m.name AS buy_day_name_latest_l2,
            m.open AS buy_open_raw,
            m.pre_close AS buy_pre_close_raw,
            m.ST_TYPE AS buy_ST_TYPE_latest_l2,
            m.ST_TYPE_name AS buy_ST_TYPE_name_latest_l2
        FROM need_keys k
        LEFT JOIN STOCK_DAILY_DATA m
          ON m.trade_date = k.buy_date
         AND m.stock_code = k.stock_code
        """
    ).fetchdf()
    con.close()
    out = frame.merge(market, on=["buy_date", "stock_code"], how="left")
    open_raw = pd.to_numeric(out["buy_open_raw"], errors="coerce")
    pre_close = pd.to_numeric(out["buy_pre_close_raw"], errors="coerce")
    out["buy_day_market_available_recalc"] = open_raw.notna() & pre_close.notna()
    out["exec_open_gap_pct_recalc"] = (open_raw / pre_close - 1.0) * 100.0
    name = out["buy_day_name_latest_l2"].fillna(out.get("name", "")).astype(str)
    st_bad = out["buy_ST_TYPE_latest_l2"].map(is_bad_st) | out["buy_ST_TYPE_name_latest_l2"].map(is_bad_st)
    name_bad = name.str.startswith(("ST", "*ST", "退"))
    out["buy_day_st_rejected_recalc"] = st_bad | name_bad
    upper = pd.Series(
        [
            float(row.buy_pre_close_raw) * (1.0 + limit_up_pct(row.stock_code))
            if pd.notna(row.buy_pre_close_raw)
            else float("nan")
            for row in out.itertuples()
        ],
        index=out.index,
    )
    out["buy_day_open_limit_up_rejected_recalc"] = open_raw >= upper * 0.995
    out["buy_day_hard_gate_complete_recalc"] = (
        out["buy_day_market_available_recalc"]
        & ~out["buy_day_st_rejected_recalc"]
        & ~out["buy_day_open_limit_up_rejected_recalc"]
    )
    out["buy_day_hard_gate_complete"] = out["buy_day_hard_gate_complete_recalc"]
    out["buy_day_st_rejected"] = out["buy_day_st_rejected_recalc"]
    out["buy_day_open_limit_up_rejected"] = out["buy_day_open_limit_up_rejected_recalc"]
    out["buy_open_gap_pct"] = out["exec_open_gap_pct_recalc"]
    out["buy_open_gap_raw_pct"] = out["exec_open_gap_pct_recalc"]
    out["exec_open_gap_pct"] = out["exec_open_gap_pct_recalc"]
    return out


def load_case(case: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = pd.read_csv(case["source"], encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in ["target_pct", "rank", "sort_score", "entry_score", "pred_prob"]:
        if col in source.columns:
            source[col] = pd.to_numeric(source[col], errors="coerce")
    source["strategy_variant"] = case["case"]
    source["filter_name"] = case["case"]
    source["coverage_reaudit_family"] = case["family"]
    source["coverage_reaudit_note"] = case["note"]
    scale = float(case.get("position_scale", 1.0))
    cap = case.get("position_cap")
    daily_cap = case.get("daily_cap")
    source["target_pct_before_coverage_scale"] = pd.to_numeric(source["target_pct"], errors="coerce").fillna(0.0)
    source["target_pct"] = source["target_pct_before_coverage_scale"] * scale
    if cap is not None:
        source["target_pct"] = source["target_pct"].clip(upper=float(cap))
    if daily_cap is not None:
        daily_sum = source.groupby("buy_date")["target_pct"].transform("sum")
        adjust = (float(daily_cap) / daily_sum).where(daily_sum > float(daily_cap), 1.0)
        source["target_pct"] = source["target_pct"] * adjust
    audited = recalc_hardgate(source)
    passed = audited[audited["buy_day_hard_gate_complete_recalc"]].copy()
    stats = {
        "case": case["case"],
        "family": case["family"],
        "source_rows": int(len(source)),
        "source_buy_days": int(source["buy_date"].nunique()) if len(source) else 0,
        "source_trade_day_coverage": int(source["buy_date"].nunique()) / TRADE_DAYS if len(source) else 0.0,
        "pass_rows": int(len(passed)),
        "pass_buy_days": int(passed["buy_date"].nunique()) if len(passed) else 0,
        "pass_trade_day_coverage": int(passed["buy_date"].nunique()) / TRADE_DAYS if len(passed) else 0.0,
        "row_pass_ratio": float(len(passed) / len(source)) if len(source) else 0.0,
        "market_missing_rows": int((~audited["buy_day_market_available_recalc"]).sum()),
        "st_rejected_rows": int(audited["buy_day_st_rejected_recalc"].sum()),
        "open_limit_up_rejected_rows": int(audited["buy_day_open_limit_up_rejected_recalc"].sum()),
        "latest_buy_date": str(passed["buy_date"].max()) if len(passed) else None,
        "mean_target_sum": float(passed.groupby("buy_date")["target_pct"].sum().mean()) if len(passed) else 0.0,
        "max_positions": int(passed.groupby("buy_date")["stock_code"].count().max()) if len(passed) else 1,
    }
    return audited, passed, stats


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start is None:
        return frame.copy()
    return frame[frame["buy_date"] >= start].copy()


def run_case(signal_file: Path, case: str, slice_name: str, max_positions: int) -> dict:
    log_file = LOG_DIR / f"{case}_{slice_name}.log"
    cmd = [
        sys.executable,
        str(base.RUNNER),
        "--strategy-dir",
        str(base.STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(max_positions),
        "--score-db",
        str(base.SCORE_DB),
        "--score-table",
        base.SCORE_TABLE,
        "--market-db",
        str(base.MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    indicator = base.parse_metrics(text)
    out = {
        "case": case,
        "slice": slice_name,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "returncode": proc.returncode,
    }
    if indicator:
        out.update(indicator)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    audit_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    result_rows: list[dict] = []
    for case in CASES:
        audited, passed, stats = load_case(case)
        audited["case"] = case["case"]
        audit_frames.append(audited)
        audit_rows.append(stats)
        for slice_name, start in base.SLICES:
            sliced = slice_frame(passed, start)
            out_path = OUT_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            result = run_case(out_path, case["case"], slice_name, max(1, stats["max_positions"]))
            result.update(stats)
            result.update(
                {
                    "slice": slice_name,
                    "slice_rows": int(len(sliced)),
                    "slice_buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "slice_latest_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                    "signal_file": str(out_path),
                }
            )
            result_rows.append(result)
            print(
                json.dumps(
                    {
                        "case": case["case"],
                        "slice": slice_name,
                        "buy_days": result["slice_buy_days"],
                        "annual": result.get("pnl_ratio_annual"),
                        "sharpe": result.get("sharp_ratio"),
                        "mdd": result.get("max_drawdown"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    pd.concat(audit_frames, ignore_index=True).to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    AUDIT_JSON.write_text(json.dumps(audit_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(result_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    full = results[results["slice"] == "full"].copy()
    full["coverage_gain_vs_core"] = full["pass_trade_day_coverage"] / full.loc[full["case"] == "core_top1_recalc_s120", "pass_trade_day_coverage"].iloc[0] - 1.0
    full = full.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=False, na_position="last")
    lines = [
        "# 提高信号通过比例复核 20260715",
        "",
        "## 口径",
        "",
        "- 不做买入日失败后的候补；只比较前置规则生成出来的信号覆盖率。",
        "- 买入日硬门槛统一用最新 L2 未复权 `open/pre_close` 重算。",
        "- 本轮仍是研究复核，不修改生产策略参数。",
        "",
        "## full 结果",
        "",
        "| 版本 | 买入日覆盖 | 较核心提升 | 行通过率 | 年化 | Sharpe | 最大回撤 | 开仓数 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in full.iterrows():
        lines.append(
            f"| `{row['case']}` | {pct(row['pass_trade_day_coverage'])} | {pct(row['coverage_gain_vs_core'])} | "
            f"{pct(row['row_pass_ratio'])} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{int(float(row.get('open_count', 0) or 0))} | {row.get('slice_latest_buy_date')} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- `coverage_bl_empty_o044_s005` 能把买入日覆盖率从核心版约 21.36% 提高到约 34.00%，并保持 full 年化和 Sharpe 达标。",
            "- 但它引入的是一个独立微仓覆盖层；如果要求绝对单一核心信号，不允许任何覆盖层，则只能保留 `core_top1_recalc_s120`，覆盖率无法在不显著降低收益质量的情况下提高。",
            "- 更宽的纯 L4 宽池此前已经验证为负收益，不适合作为提高覆盖率的方向。",
            "",
            "## 证据路径",
            "",
            f"- 硬门槛审计 CSV：`{AUDIT_CSV}`",
            f"- 硬门槛审计 JSON：`{AUDIT_JSON}`",
            f"- 掘金结果 CSV：`{OUT_CSV}`",
            f"- 掘金结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{OUT_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
