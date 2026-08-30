from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "selective_edge_boost"
LOG_DIR = REPORT_DIR / "logs" / "selective_edge_boost"
OUT_CSV = REPORT_DIR / "selective_edge_boost_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "selective_edge_boost_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "hps": REPORT_DIR / "signals" / "highsharpe_position_scale" / "hps_sc092_x1040_cap100.csv",
    "sb_pred99": REPORT_DIR / "signals" / "score_bucket_weight_refine" / "sb_pred10_99_x115_else098.csv",
    "sb_bal102": REPORT_DIR / "signals" / "score_bucket_weight_refine" / "sb_balanced_edge_x102.csv",
    "ss092": REPORT_DIR / "signals" / "top3_sell_scale_refine" / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv",
}


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


CASES = [
    {
        "case": "seb_hps_pred99_deep_liq_x110",
        "source": "hps",
        "base_scale": 1.03,
        "good_mult": 1.10,
        "bad_mult": 0.90,
        "good": {"pred10": 0.99, "pred1": 0.95, "pct_max": -5.0, "amount": 500000, "atr_max": 8.0},
        "bad": {"pct_min": -3.0, "gap_min": 0.5, "atr_min": 8.0},
    },
    {
        "case": "seb_hps_pred99_deep_liq_x115",
        "source": "hps",
        "base_scale": 1.02,
        "good_mult": 1.15,
        "bad_mult": 0.88,
        "good": {"pred10": 0.99, "pred1": 0.95, "pct_max": -5.0, "amount": 500000, "atr_max": 8.0},
        "bad": {"pct_min": -3.0, "gap_min": 0.5, "atr_min": 8.0},
    },
    {
        "case": "seb_hps_pred10_995_x118",
        "source": "hps",
        "base_scale": 1.01,
        "good_mult": 1.18,
        "bad_mult": 0.92,
        "good": {"pred10": 0.995, "pred1": 0.90, "pct_max": -4.0, "amount": 300000, "atr_max": 10.0},
        "bad": {"pct_min": -2.5, "gap_min": 0.75, "atr_min": 10.0},
    },
    {
        "case": "seb_hps_deep8_pred97_x116",
        "source": "hps",
        "base_scale": 1.02,
        "good_mult": 1.16,
        "bad_mult": 0.90,
        "good": {"pred10": 0.97, "pred1": 0.90, "pct_max": -8.0, "amount": 200000, "atr_max": 12.0},
        "bad": {"pct_min": -2.5, "gap_min": 0.5, "atr_min": 12.0},
    },
    {
        "case": "seb_hps_lowatr_highscore_x112",
        "source": "hps",
        "base_scale": 1.02,
        "good_mult": 1.12,
        "bad_mult": 0.90,
        "good": {"pred10": 0.98, "pred1": 0.90, "pct_max": -4.0, "amount": 300000, "atr_max": 3.0},
        "bad": {"pct_min": -3.0, "gap_min": 0.75, "atr_min": 8.0},
    },
    {
        "case": "seb_sbpred99_trim_gap_atr_x102",
        "source": "sb_pred99",
        "base_scale": 1.02,
        "good_mult": 1.06,
        "bad_mult": 0.86,
        "good": {"pred10": 0.99, "pred1": 0.95, "pct_max": -5.0, "amount": 500000, "atr_max": 8.0},
        "bad": {"pct_min": -3.0, "gap_min": 0.5, "atr_min": 8.0},
    },
    {
        "case": "seb_sbbal102_pred99_deep_x102",
        "source": "sb_bal102",
        "base_scale": 1.02,
        "good_mult": 1.08,
        "bad_mult": 0.84,
        "good": {"pred10": 0.99, "pred1": 0.95, "pct_max": -5.0, "amount": 500000, "atr_max": 8.0},
        "bad": {"pct_min": -3.0, "gap_min": 0.5, "atr_min": 8.0},
    },
    {
        "case": "seb_ss092_aggressive_pred99_x122",
        "source": "ss092",
        "base_scale": 1.08,
        "good_mult": 1.22,
        "bad_mult": 0.82,
        "good": {"pred10": 0.99, "pred1": 0.95, "pct_max": -5.0, "amount": 500000, "atr_max": 8.0},
        "bad": {"pct_min": -3.0, "gap_min": 0.5, "atr_min": 8.0},
    },
    {
        "case": "seb_ss092_balanced_pred98_x116",
        "source": "ss092",
        "base_scale": 1.06,
        "good_mult": 1.16,
        "bad_mult": 0.86,
        "good": {"pred10": 0.98, "pred1": 0.90, "pct_max": -4.0, "amount": 300000, "atr_max": 8.0},
        "bad": {"pct_min": -2.5, "gap_min": 0.75, "atr_min": 8.0},
    },
    {
        "case": "seb_ss092_lowatr_x118",
        "source": "ss092",
        "base_scale": 1.08,
        "good_mult": 1.18,
        "bad_mult": 0.84,
        "good": {"pred10": 0.98, "pred1": 0.90, "pct_max": -4.0, "amount": 300000, "atr_max": 3.5},
        "bad": {"pct_min": -3.0, "gap_min": 0.5, "atr_min": 8.0},
    },
]


def masks(df: pd.DataFrame, case: dict) -> tuple[pd.Series, pd.Series]:
    pred10 = num(df, "pred_10d")
    pred1 = num(df, "pred_1d")
    pct = num(df, "signal_pct_chg_raw")
    amount = num(df, "amount")
    atr = num(df, "atr_qfq", default=99.0)
    gap = num(df, "buy_open_gap_raw_pct", default=999.0)
    if "buy_open_gap_pct" in df.columns:
        gap = gap.where(gap != 999.0, num(df, "buy_open_gap_pct", default=999.0))

    good_cfg = case["good"]
    bad_cfg = case["bad"]
    good = (
        (pred10 >= float(good_cfg["pred10"]))
        & (pred1 >= float(good_cfg["pred1"]))
        & (pct <= float(good_cfg["pct_max"]))
        & (amount >= float(good_cfg["amount"]))
        & (atr <= float(good_cfg["atr_max"]))
    )
    bad = (
        (pct > float(bad_cfg["pct_min"]))
        | (gap > float(bad_cfg["gap_min"]))
        | (atr >= float(bad_cfg["atr_min"]))
    )
    return good, bad


def write_variant(case: dict) -> dict:
    src = SOURCES[case["source"]]
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    good, bad = masks(df, case)
    target = num(df, "target_pct") * float(case["base_scale"])
    mult = pd.Series(1.0, index=df.index, dtype="float64")
    mult.loc[good] *= float(case["good_mult"])
    mult.loc[bad] *= float(case["bad_mult"])
    target = target * mult
    daily_sum = target.groupby(df["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = target * cap_scale
    df = df[df["target_pct"] > 0].copy()
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["selective_good_flag"] = good.astype(int)
    df["selective_bad_flag"] = bad.astype(int)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "source": case["source"],
        "source_file": str(src),
        "signal_file": str(out),
        "base_scale": float(case["base_scale"]),
        "good_mult": float(case["good_mult"]),
        "bad_mult": float(case["bad_mult"]),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "good_rows": int(good.sum()),
        "bad_rows": int(bad.sum()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()) if len(df) else 0.0,
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()) if len(df) else 0.0,
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

    manifest = [write_variant(case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        results.append(result)
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

    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
