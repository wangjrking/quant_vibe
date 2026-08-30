from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SIGNAL_DIR = REPORT_DIR / "signals" / "gap_nan_fill_refine"
LOG_DIR = REPORT_DIR / "logs" / "gap_nan_fill_refine"
OUT_CSV = REPORT_DIR / "gap_nan_fill_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "gap_nan_fill_refine_juejin_results_20260714.json"

SOURCES = {
    "ge_h69": REPORT_DIR
    / "signals"
    / "top3_pick1_gap_edge_refine"
    / "ge_h69_m36_l27_ss80_pg35_ds100_dg110.csv",
    "sc092": REPORT_DIR
    / "signals"
    / "top3_sell_scale_refine"
    / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv",
}

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


def load_gap_frame(codes: list[str], dates: list[str]) -> pd.DataFrame:
    if not codes or not dates:
        return pd.DataFrame(columns=["stock_code", "buy_date", "l2_open_gap_pct"])
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        df = con.execute(
            """
            SELECT stock_code, trade_date AS buy_date,
                   CASE
                       WHEN pre_close IS NOT NULL AND pre_close != 0 AND open IS NOT NULL
                       THEN (open / pre_close - 1.0) * 100.0
                       ELSE NULL
                   END AS l2_open_gap_pct
            FROM STOCK_DAILY_DATA
            WHERE stock_code IN (SELECT * FROM UNNEST(?))
              AND trade_date IN (SELECT * FROM UNNEST(?))
            """,
            [codes, dates],
        ).fetchdf()
    finally:
        con.close()
    df["buy_date"] = df["buy_date"].astype(str)
    return df


def apply_fill(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["buy_date"] = out["buy_date"].astype(str)
    out["stock_code"] = out["stock_code"].astype(str)
    gap = pd.to_numeric(out.get("buy_open_gap_pct"), errors="coerce")
    gap_raw = pd.to_numeric(out.get("buy_open_gap_raw_pct"), errors="coerce") if "buy_open_gap_raw_pct" in out else gap
    current = gap_raw.fillna(gap)

    gap_df = load_gap_frame(out["stock_code"].dropna().unique().tolist(), out["buy_date"].dropna().unique().tolist())
    out = out.merge(gap_df, on=["stock_code", "buy_date"], how="left")
    l2_gap = pd.to_numeric(out["l2_open_gap_pct"], errors="coerce")
    out["gap_was_missing"] = current.isna()

    if mode == "l2_fill_else_neutral":
        filled = current.fillna(l2_gap).fillna(0.0)
    elif mode == "l2_fill_else_keep_penalty":
        filled = current.fillna(l2_gap)
    elif mode == "neutral_missing":
        filled = current.fillna(0.0)
    else:
        raise ValueError(mode)
    out["exec_open_gap_pct"] = filled
    out["buy_open_gap_pct"] = filled
    out["buy_open_gap_raw_pct"] = filled
    return out


def recompute_ge_targets(df: pd.DataFrame, case: dict) -> pd.DataFrame:
    out = df.copy()
    bucket = pd.to_numeric(out["quality_bucket"], errors="coerce").fillna(0).astype(int)
    pct = pd.to_numeric(out["signal_pct_chg_raw"], errors="coerce").fillna(0.0)
    gap = pd.to_numeric(out["exec_open_gap_pct"], errors="coerce")
    target = pd.Series(case["low"], index=out.index, dtype=float)
    target.loc[bucket == 1] = case["mid"]
    target.loc[bucket == 2] = case["high"]
    target.loc[pct > -3.0] *= case["shallow_scale"]
    target.loc[pct <= -5.0] *= case["deep_scale"]
    target.loc[gap > 0.5] *= case["pos_gap_scale"]
    target.loc[gap <= -3.0] *= case["deep_gap_scale"]
    target = target.clip(lower=0.0, upper=case["cap"])
    out["target_pct"] = target
    out = out[out["target_pct"] > 0].copy()
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    return out


CASES = [
    {
        "name": "ge_h69_l2fill_neutral",
        "source": "ge_h69",
        "mode": "l2_fill_else_neutral",
        "recompute_ge": True,
        "high": 0.69,
        "mid": 0.36,
        "low": 0.27,
        "shallow_scale": 0.80,
        "deep_scale": 1.00,
        "pos_gap_scale": 0.35,
        "deep_gap_scale": 1.10,
        "cap": 0.91,
    },
    {
        "name": "ge_h69_l2fill_keep",
        "source": "ge_h69",
        "mode": "l2_fill_else_keep_penalty",
        "recompute_ge": True,
        "high": 0.69,
        "mid": 0.36,
        "low": 0.27,
        "shallow_scale": 0.80,
        "deep_scale": 1.00,
        "pos_gap_scale": 0.35,
        "deep_gap_scale": 1.10,
        "cap": 0.91,
    },
    {"name": "sc092_l2fill_neutral_x100", "source": "sc092", "mode": "l2_fill_else_neutral", "target_scale": 1.00},
    {"name": "sc092_l2fill_neutral_x104", "source": "sc092", "mode": "l2_fill_else_neutral", "target_scale": 1.04},
    {"name": "sc092_l2fill_keep_x104", "source": "sc092", "mode": "l2_fill_else_keep_penalty", "target_scale": 1.04},
]


def write_variant(case: dict) -> dict:
    df = pd.read_csv(SOURCES[case["source"]], encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df = apply_fill(df, case["mode"])
    if case.get("recompute_ge"):
        df = recompute_ge_targets(df, case)
    else:
        scale = float(case.get("target_scale", 1.0))
        target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0) * scale
        daily_sum = target.groupby(df["buy_date"]).transform("sum")
        cap_scale = (1.0 / daily_sum).clip(upper=1.0)
        df["target_pct"] = target * cap_scale
        df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
    out = SIGNAL_DIR / f"{case['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["name"],
        "source": case["source"],
        "mode": case["mode"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "gap_was_missing_rows": int(df["gap_was_missing"].sum()) if "gap_was_missing" in df else 0,
        "gap_after_missing_rows": int(pd.to_numeric(df["exec_open_gap_pct"], errors="coerce").isna().sum()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        row = write_variant(case)
        result = base_mod.run_juejin(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
        ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    hits = [
        row
        for row in results
        if (row.get("pnl_ratio_annual") or 0) >= 5.0
        and (row.get("sharp_ratio") or 0) >= 4.0
        and (row.get("max_drawdown") or 1) <= 0.40
    ]
    summary = {
        "result_csv": str(OUT_CSV),
        "result_json": str(OUT_JSON),
        "cases": len(results),
        "target_hits": len(hits),
        "target_hits_table": sorted(
            hits,
            key=lambda row: (row.get("sharp_ratio") or -999, row.get("pnl_ratio_annual") or -999),
            reverse=True,
        ),
        "results": results,
    }
    (REPORT_DIR / "gap_nan_fill_refine_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
