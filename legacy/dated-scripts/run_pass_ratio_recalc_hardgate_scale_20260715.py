from __future__ import annotations

import importlib.util
import json
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

SOURCE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "signals"
    / "open_gap_deep_rebalance"
    / "ogd_gap1_x050.csv"
)
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT_CSV = base.REPORT_DIR / "pass_ratio_recalc_hardgate_scale_juejin_20260715.csv"
OUT_JSON = base.REPORT_DIR / "pass_ratio_recalc_hardgate_scale_juejin_20260715.json"
OUT_MD = base.REPORT_DIR / "pass_ratio_recalc_hardgate_scale_review_20260715.md"
SIGNAL_DIR = base.REPORT_DIR / "signals" / "recalc_hardgate_scale"
LOG_DIR = base.REPORT_DIR / "logs" / "recalc_hardgate_scale_20260715"
AUDIT_JSON = base.REPORT_DIR / "recalc_hardgate_top1_audit_20260715.json"
AUDIT_CSV = base.REPORT_DIR / "recalc_hardgate_top1_audit_20260715.csv"

CASES = [
    {"case": "rh_top1_base", "scale": 1.00, "cap": 0.65},
    {"case": "rh_top1_s105_cap065", "scale": 1.05, "cap": 0.65},
    {"case": "rh_top1_s110_cap065", "scale": 1.10, "cap": 0.65},
    {"case": "rh_top1_s115_cap065", "scale": 1.15, "cap": 0.65},
    {"case": "rh_top1_s120_cap065", "scale": 1.20, "cap": 0.65},
    {"case": "rh_top1_s125_cap065", "scale": 1.25, "cap": 0.65},
    {"case": "rh_top1_s130_cap065", "scale": 1.30, "cap": 0.65},
    {"case": "rh_top1_s138_cap065", "scale": 1.38, "cap": 0.65},
    {"case": "rh_top1_s142_cap065", "scale": 1.42, "cap": 0.65},
    {"case": "rh_top1_s145_cap065", "scale": 1.45, "cap": 0.65},
    {"case": "rh_top1_s148_cap065", "scale": 1.48, "cap": 0.65},
    {"case": "rh_top1_s142_cap067", "scale": 1.42, "cap": 0.67},
]


def limit_up_pct(stock_code: str) -> float:
    code = str(stock_code or "").split(".", 1)[0]
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def is_bad_st(value: object) -> bool:
    text = str(value or "").strip()
    return text not in {"", "0", "0.0", "None", "nan", "NaN", "正常", "无"}


def build_intended_top1(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    for col in ["target_pct", "sort_score", "rank", "pred_prob", "entry_score"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    sort_cols = [col for col in ["buy_date", "sort_score", "target_pct", "pred_prob", "entry_score", "stock_code"] if col in frame.columns]
    ascending = [True] + [False] * (len(sort_cols) - 2) + [True] if len(sort_cols) >= 2 else True
    frame = frame.sort_values(sort_cols, ascending=ascending)
    return frame.groupby("buy_date", as_index=False).head(1).copy()


def recalc_hardgate(top1: pd.DataFrame) -> pd.DataFrame:
    keys = top1[["buy_date", "stock_code"]].drop_duplicates().copy()
    con = duckdb.connect(str(L2_DB), read_only=True)
    con.register("need_keys", keys)
    market = con.execute(
        """
        SELECT
            k.buy_date,
            k.stock_code,
            m.name AS buy_day_name,
            m.open AS buy_open_raw,
            m.pre_close AS buy_pre_close_raw,
            m.ST_TYPE AS buy_ST_TYPE,
            m.ST_TYPE_name AS buy_ST_TYPE_name
        FROM need_keys k
        LEFT JOIN STOCK_DAILY_DATA m
          ON m.trade_date = k.buy_date
         AND m.stock_code = k.stock_code
        """
    ).fetchdf()
    con.close()
    out = top1.merge(market, on=["buy_date", "stock_code"], how="left")
    out["buy_day_market_available_recalc"] = out["buy_open_raw"].notna() & out["buy_pre_close_raw"].notna()
    out["exec_open_gap_pct_recalc"] = (pd.to_numeric(out["buy_open_raw"], errors="coerce") / pd.to_numeric(out["buy_pre_close_raw"], errors="coerce") - 1.0) * 100.0
    st_type_bad = out["buy_ST_TYPE"].map(is_bad_st)
    st_name_bad = out["buy_ST_TYPE_name"].map(is_bad_st)
    name_text = out["buy_day_name"].fillna(out.get("name", "")).astype(str)
    name_bad = name_text.str.startswith(("ST", "*ST", "退"))
    out["buy_day_st_rejected_recalc"] = st_type_bad | st_name_bad | name_bad
    upper = pd.Series([float(row.buy_pre_close_raw) * (1.0 + limit_up_pct(row.stock_code)) if pd.notna(row.buy_pre_close_raw) else float("nan") for row in out.itertuples()], index=out.index)
    out["buy_day_open_limit_up_rejected_recalc"] = pd.to_numeric(out["buy_open_raw"], errors="coerce") >= upper * 0.995
    out["buy_day_hard_gate_complete_recalc"] = (
        out["buy_day_market_available_recalc"]
        & ~out["buy_day_st_rejected_recalc"]
        & ~out["buy_day_open_limit_up_rejected_recalc"]
    )
    out["exec_open_gap_pct"] = out["exec_open_gap_pct_recalc"]
    out["buy_open_gap_pct"] = out["exec_open_gap_pct_recalc"]
    out["buy_open_gap_raw_pct"] = out["exec_open_gap_pct_recalc"]
    out["buy_day_hard_gate_complete"] = out["buy_day_hard_gate_complete_recalc"]
    out["buy_day_st_rejected"] = out["buy_day_st_rejected_recalc"]
    out["buy_day_open_limit_up_rejected"] = out["buy_day_open_limit_up_rejected_recalc"]
    return out


def scale_case(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    out = source.copy()
    raw = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0) * float(case["scale"])
    out["target_pct_before_recalc_scale"] = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0)
    out["target_pct"] = raw.clip(upper=float(case["cap"]))
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    out["recalc_hardgate_scale"] = float(case["scale"])
    out["recalc_hardgate_cap"] = float(case["cap"])
    return out


def run_case(signal_file: Path, case: str, slice_name: str) -> dict:
    old_log_dir = base.LOG_DIR
    base.LOG_DIR = LOG_DIR
    try:
        return base.run_juejin(signal_file, case, slice_name)
    finally:
        base.LOG_DIR = old_log_dir


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    top1 = build_intended_top1(source)
    recalced = recalc_hardgate(top1)
    recalced.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    passed = recalced[recalced["buy_day_hard_gate_complete_recalc"]].copy()
    audit = {
        "source_rows": int(len(source)),
        "source_buy_days": int(source["buy_date"].nunique()),
        "top1_rows": int(len(top1)),
        "top1_min_buy_date": str(top1["buy_date"].min()),
        "top1_max_buy_date": str(top1["buy_date"].max()),
        "recalc_pass_rows": int(len(passed)),
        "recalc_pass_min_buy_date": str(passed["buy_date"].min()) if len(passed) else None,
        "recalc_pass_max_buy_date": str(passed["buy_date"].max()) if len(passed) else None,
        "market_missing_rows": int((~recalced["buy_day_market_available_recalc"]).sum()),
        "st_rejected_rows": int(recalced["buy_day_st_rejected_recalc"].sum()),
        "open_limit_up_rejected_rows": int(recalced["buy_day_open_limit_up_rejected_recalc"].sum()),
    }
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for case in CASES:
        built = scale_case(passed, case)
        for slice_name, start in base.SLICES:
            sliced = base.slice_frame(built, start)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_case(out_path, case["case"], slice_name)
            result.update(
                {
                    "scale": float(case["scale"]),
                    "cap": float(case["cap"]),
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "max_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                }
            )
            open_count = result.get("open_count")
            result["row_execution_ratio"] = float(open_count) / float(len(sliced)) if open_count is not None and len(sliced) else None
            rows.append(result)
            print(json.dumps({"case": result["case"], "slice": slice_name, "rows": len(sliced), "max_buy": result["max_buy_date"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    grouped = []
    for case, group in frame.groupby("case"):
        item = {"case": case}
        for sl in ["full", "from_202501", "recent60"]:
            row = group[group["slice"] == sl].iloc[0]
            item[f"{sl}_annual"] = row.get("pnl_ratio_annual")
            item[f"{sl}_sharpe"] = row.get("sharp_ratio")
            item[f"{sl}_mdd"] = row.get("max_drawdown")
            item[f"{sl}_max_buy"] = row.get("max_buy_date")
        grouped.append(item)
    gframe = pd.DataFrame(grouped)
    gframe["min_annual"] = gframe[["full_annual", "from_202501_annual", "recent60_annual"]].min(axis=1)
    gframe["min_sharpe"] = gframe[["full_sharpe", "from_202501_sharpe", "recent60_sharpe"]].min(axis=1)
    gframe = gframe.sort_values(["full_sharpe", "full_annual"], ascending=False)

    lines = [
        "# 最新 L2 重算硬门槛 Top1 调仓掘金复跑 20260715",
        "",
        "## 硬门槛重算结论",
        "",
        f"- 原始信号行数：{audit['source_rows']}，原始买入日：{audit['source_buy_days']}。",
        f"- 每日 Top1 行数：{audit['top1_rows']}，覆盖买入日：{audit['top1_min_buy_date']} 到 {audit['top1_max_buy_date']}。",
        f"- 用最新 L2 未复权 open/pre_close 重算后，通过行数：{audit['recalc_pass_rows']}，覆盖买入日：{audit['recalc_pass_min_buy_date']} 到 {audit['recalc_pass_max_buy_date']}。",
        f"- 市场缺失：{audit['market_missing_rows']}；ST/风险警示剔除：{audit['st_rejected_rows']}；开盘涨停剔除：{audit['open_limit_up_rejected_rows']}。",
        "",
        "## 掘金复跑结果",
        "",
        "| 版本 | full年化 | full Sharpe | full回撤 | 2025以来年化 | recent60年化 | recent60 max_buy | 三段最小年化 | 三段最小Sharpe |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in gframe.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {base.pct(row['full_annual'])} | {float(row['full_sharpe']):.4f} | "
            f"{base.pct(row['full_mdd'])} | {base.pct(row['from_202501_annual'])} | "
            f"{base.pct(row['recent60_annual'])} | {row['recent60_max_buy']} | "
            f"{base.pct(row['min_annual'])} | {float(row['min_sharpe']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 硬门槛审计 JSON：`{AUDIT_JSON}`",
            f"- 硬门槛审计 CSV：`{AUDIT_CSV}`",
            f"- 结果 CSV：`{OUT_CSV}`",
            f"- 结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
