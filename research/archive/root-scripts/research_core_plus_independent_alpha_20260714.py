from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_plus_independent_alpha"
LOG_DIR = REPORT_DIR / "logs" / "core_plus_independent_alpha"
OUT_CSV = REPORT_DIR / "core_plus_independent_alpha_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_plus_independent_alpha_juejin_results_20260714.json"
SUMMARY_JSON = REPORT_DIR / "core_plus_independent_alpha_summary_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CORE_FILES = {
    "core_x115": REPORT_DIR / "signals" / "market_regime_scaling" / "mkt_panic_deep_x115.csv",
    "core_x125": REPORT_DIR / "signals" / "market_regime_scaling" / "mkt_panic_deep_x125.csv",
}
OVERLAY_FILES = {
    "iap04404": REPORT_DIR / "signals" / "independent_alpha_pockets" / "iap_04404_p10p1_broad_t2_h2.csv",
    "iap02868": REPORT_DIR / "signals" / "independent_alpha_pockets" / "iap_02868_p10p1_broad_t1_h2.csv",
    "iap04340": REPORT_DIR / "signals" / "independent_alpha_pockets" / "iap_04340_p10p1_broad_t2_h2.csv",
}


CASES = [
    {"case": "c115_empty_o044_s010_cap096", "core": "core_x115", "overlay": ["iap04404"], "mode": "empty", "scale": 0.10, "cap": 0.96, "max_overlay": 1},
    {"case": "c115_empty_o044_s015_cap096", "core": "core_x115", "overlay": ["iap04404"], "mode": "empty", "scale": 0.15, "cap": 0.96, "max_overlay": 1},
    {"case": "c115_lowsum_o044_s010_cap096", "core": "core_x115", "overlay": ["iap04404"], "mode": "lowsum", "scale": 0.10, "cap": 0.96, "max_overlay": 1, "low_sum": 0.45},
    {"case": "c115_lowsum_o044_s015_cap096", "core": "core_x115", "overlay": ["iap04404"], "mode": "lowsum", "scale": 0.15, "cap": 0.96, "max_overlay": 1, "low_sum": 0.45},
    {"case": "c115_empty_o028_s010_cap096", "core": "core_x115", "overlay": ["iap02868"], "mode": "empty", "scale": 0.10, "cap": 0.96, "max_overlay": 1},
    {"case": "c115_empty_o043_s010_cap096", "core": "core_x115", "overlay": ["iap04340"], "mode": "empty", "scale": 0.10, "cap": 0.96, "max_overlay": 1},
    {"case": "c115_empty_o3_s008_cap096", "core": "core_x115", "overlay": ["iap04404", "iap02868", "iap04340"], "mode": "empty", "scale": 0.08, "cap": 0.96, "max_overlay": 2},
    {"case": "c115_lowsum_o3_s008_cap096", "core": "core_x115", "overlay": ["iap04404", "iap02868", "iap04340"], "mode": "lowsum", "scale": 0.08, "cap": 0.96, "max_overlay": 2, "low_sum": 0.45},
    {"case": "c125_empty_o044_s010_cap096", "core": "core_x125", "overlay": ["iap04404"], "mode": "empty", "scale": 0.10, "cap": 0.96, "max_overlay": 1},
    {"case": "c125_lowsum_o044_s010_cap096", "core": "core_x125", "overlay": ["iap04404"], "mode": "lowsum", "scale": 0.10, "cap": 0.96, "max_overlay": 1, "low_sum": 0.45},
]


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    df["sort_score"] = pd.to_numeric(df.get("sort_score", df.get("pred_prob")), errors="coerce").fillna(0.0)
    return df


def build_case(case: dict) -> dict:
    core = load_signal(CORE_FILES[case["core"]]).copy()
    core["layer"] = "core"
    overlays = []
    for name in case["overlay"]:
        odf = load_signal(OVERLAY_FILES[name]).copy()
        odf["layer"] = name
        overlays.append(odf)
    overlay = pd.concat(overlays, ignore_index=True)
    core_sum = core.groupby("buy_date")["target_pct"].sum().rename("core_sum")
    overlay = overlay.merge(core_sum, on="buy_date", how="left")
    overlay["core_sum"] = overlay["core_sum"].fillna(0.0)
    if case["mode"] == "empty":
        overlay = overlay[overlay["core_sum"] <= 0].copy()
    elif case["mode"] == "lowsum":
        overlay = overlay[overlay["core_sum"] < float(case.get("low_sum", 0.45))].copy()
    else:
        raise ValueError(case["mode"])
    if not overlay.empty:
        core_keys = set(zip(core["buy_date"], core["stock_code"]))
        overlay = overlay[~overlay.apply(lambda r: (r["buy_date"], r["stock_code"]) in core_keys, axis=1)].copy()
        overlay = (
            overlay.sort_values(["buy_date", "sort_score"], ascending=[True, False])
            .groupby("buy_date", group_keys=False)
            .head(int(case["max_overlay"]))
            .copy()
        )
        overlay["target_pct"] = overlay["target_pct"] * float(case["scale"])
        overlay["strategy_variant"] = case["case"]
        overlay["source_strategy_variant"] = overlay["source_strategy_variant"].astype(str) + "_overlay"
        overlay["filter_name"] = case["case"]
    df = pd.concat([core, overlay], ignore_index=True, sort=False)
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    df["target_pct"] = df["target_pct"] * (float(case["cap"]) / daily_sum).clip(upper=1.0)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["rank"] = df.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "core_rows": int(len(core)),
        "overlay_rows": int(len(overlay)),
        "buy_days": int(df["buy_date"].nunique()),
        "overlay_buy_days": int(overlay["buy_date"].nunique()) if len(overlay) else 0,
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        "max_positions": int(max(df.groupby("buy_date")["stock_code"].count().max(), 1)),
        "holding_days": 2,
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_mp{row['max_positions']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(base_mod.RUNNER),
        "--strategy-dir", str(base_mod.STRATEGY_DIR),
        "--signal-file", row["signal_file"],
        "--log-file", str(log),
        "--score-db", str(base_mod.SCORE_DB),
        "--score-table", base_mod.SCORE_TABLE,
        "--market-db", str(base_mod.MARKET_DB),
        "--max-positions", str(row["max_positions"]),
        "--holding-days", str(row["holding_days"]),
        "--max-holding-days", "3",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    ind = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            ind = payload.get("indicator") or {}
            if isinstance(ind, str):
                ind = base_mod.parse_indicator_text(ind)
        except Exception:
            ind = base_mod.extract_indicator(text)
    else:
        ind = base_mod.extract_indicator(text)
    out = dict(row)
    out["returncode"] = proc.returncode
    out["log_file"] = str(log)
    if isinstance(ind, dict):
        out.update(ind)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifests = [build_case(case) for case in CASES]
    results = []
    for row in manifests:
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"case": result.get("case"), "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "open": result.get("open_count"), "overlay_rows": result.get("overlay_rows")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    summary = {
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
