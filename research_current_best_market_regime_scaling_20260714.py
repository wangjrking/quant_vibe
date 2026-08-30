from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE = REPORT_DIR / "signals" / "current_best_stability_guards" / "guard_target_cap068.csv"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SIGNAL_DIR = REPORT_DIR / "signals" / "market_regime_scaling"
LOG_DIR = REPORT_DIR / "logs" / "market_regime_scaling"
OUT_CSV = REPORT_DIR / "market_regime_scaling_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "market_regime_scaling_juejin_results_20260714.json"
FEATURE_CSV = REPORT_DIR / "market_regime_features_20260714.csv"
SUMMARY_JSON = REPORT_DIR / "market_regime_scaling_summary_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"case": "mkt_bad_x085", "bad_scale": 0.85},
    {"case": "mkt_bad_x070", "bad_scale": 0.70},
    {"case": "mkt_bad_x055", "bad_scale": 0.55},
    {"case": "mkt_severe_x070", "severe_scale": 0.70},
    {"case": "mkt_severe_x055", "severe_scale": 0.55},
    {"case": "mkt_bad_x085_good_x105", "bad_scale": 0.85, "good_scale": 1.05},
    {"case": "mkt_bad_x070_good_x110", "bad_scale": 0.70, "good_scale": 1.10},
    {"case": "mkt_panic_deep_x115", "panic_deep_scale": 1.15},
    {"case": "mkt_panic_deep_x125", "panic_deep_scale": 1.25},
    {"case": "mkt_amount_low_x080", "amount_low_scale": 0.80},
    {"case": "mkt_amount_low_x070_good_x105", "amount_low_scale": 0.70, "good_scale": 1.05},
    {"case": "mkt_bad_or_amount_low_x070", "bad_scale": 0.70, "amount_low_scale": 0.70},
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in ["target_pct", "signal_pct_chg_raw", "pred_10d", "quality_bucket"]:
        if col in df.columns:
            df[col] = _num(df, col)
    return df


def build_market_features(source: pd.DataFrame) -> pd.DataFrame:
    min_date = str(source["signal_date"].min())
    max_date = str(source["signal_date"].max())
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        features = con.execute(
            """
            select
              trade_date,
              avg(pct_chg) as mkt_avg_pct,
              avg(case when pct_chg > 0 then 1.0 else 0.0 end) as mkt_up_ratio,
              avg(case when pct_chg <= -5 then 1.0 else 0.0 end) as mkt_down5_ratio,
              avg(case when pct_chg >= 9.5 then 1.0 else 0.0 end) as mkt_limitup_like_ratio,
              sum(amount) as mkt_amount_sum,
              count(*) as mkt_stock_count,
              max(index_2000_close / nullif(index_2000_open, 0) - 1.0) * 100.0 as index_2000_oc_pct
            from STOCK_DAILY_DATA
            where trade_date between ? and ?
              and stock_code not like '%.BJ'
              and pct_chg is not null
              and amount is not null
            group by trade_date
            order by trade_date
            """,
            [min_date, max_date],
        ).fetchdf()
    finally:
        con.close()
    features["trade_date"] = features["trade_date"].astype(str)
    features = features.sort_values("trade_date").reset_index(drop=True)
    features["mkt_avg_pct_5d"] = features["mkt_avg_pct"].rolling(5, min_periods=1).mean()
    features["mkt_up_ratio_5d"] = features["mkt_up_ratio"].rolling(5, min_periods=1).mean()
    features["mkt_amount_sum_20d"] = features["mkt_amount_sum"].rolling(20, min_periods=5).mean()
    features["mkt_amount_ratio_20d"] = features["mkt_amount_sum"] / features["mkt_amount_sum_20d"]
    features.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")
    return features


def add_market_flags(source: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = source.merge(features, left_on="signal_date", right_on="trade_date", how="left")
    out["mkt_bad"] = (
        (out["mkt_up_ratio"] < 0.35)
        | (out["mkt_avg_pct"] < -1.0)
        | (out["mkt_down5_ratio"] > 0.10)
        | (out["mkt_up_ratio_5d"] < 0.38)
    )
    out["mkt_severe"] = (
        (out["mkt_up_ratio"] < 0.25)
        | (out["mkt_avg_pct"] < -2.0)
        | (out["mkt_down5_ratio"] > 0.18)
    )
    out["mkt_good"] = (
        (out["mkt_up_ratio"] >= 0.45)
        & (out["mkt_avg_pct"] > -0.35)
        & (out["mkt_down5_ratio"] < 0.06)
    )
    out["mkt_amount_low"] = out["mkt_amount_ratio_20d"] < 0.82
    out["mkt_panic_deep"] = out["mkt_severe"] & (out["signal_pct_chg_raw"] <= -8.0) & (out["pred_10d"] >= 0.96)
    return out


def write_variant(source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    scale = pd.Series(1.0, index=df.index)
    if "bad_scale" in case:
        scale = scale.mask(df["mkt_bad"].fillna(False), scale * float(case["bad_scale"]))
    if "severe_scale" in case:
        scale = scale.mask(df["mkt_severe"].fillna(False), scale * float(case["severe_scale"]))
    if "amount_low_scale" in case:
        scale = scale.mask(df["mkt_amount_low"].fillna(False), scale * float(case["amount_low_scale"]))
    if "good_scale" in case:
        scale = scale.mask(df["mkt_good"].fillna(False), scale * float(case["good_scale"]))
    if "panic_deep_scale" in case:
        scale = scale.mask(df["mkt_panic_deep"].fillna(False), scale * float(case["panic_deep_scale"]))

    df["target_pct"] = (target * scale).clip(upper=0.68)
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    df["target_pct"] = df["target_pct"] * (1.0 / daily_sum).clip(upper=1.0)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["market_regime_scale"] = scale
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "bad_rows": int(df["mkt_bad"].fillna(False).sum()),
        "severe_rows": int(df["mkt_severe"].fillna(False).sum()),
        "good_rows": int(df["mkt_good"].fillna(False).sum()),
        "amount_low_rows": int(df["mkt_amount_low"].fillna(False).sum()),
        "panic_deep_rows": int(df["mkt_panic_deep"].fillna(False).sum()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        **{k: v for k, v in case.items() if k != "case"},
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = load_source()
    features = build_market_features(source)
    source = add_market_flags(source, features)
    manifest = [write_variant(source, case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
        ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    summary = {
        "source": str(SOURCE),
        "feature_csv": str(FEATURE_CSV),
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best_by_sharpe": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(8).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
