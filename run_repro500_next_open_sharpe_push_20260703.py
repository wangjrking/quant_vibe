from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_SIGNAL = (
    ROOT
    / "quant/main/strategy_library/production/prod_repro500_p435_s96_p10d70_v20260702/signals/full_history_repro500_p435_s96_p10d70.csv"
)
ACTIVE_WIDE = REPORT_DIR / "active_l4_wide_cache.duckdb"
OUT_DIR = REPORT_DIR / "repro500_next_open_sharpe_push"
SIGNAL_DIR = OUT_DIR / "signals"
LOG_DIR = OUT_DIR / "juejin_runs"
LOCAL_SUMMARY = OUT_DIR / "local_proxy_summary.csv"
JUEJIN_SUMMARY = OUT_DIR / "juejin_summary.csv"

STRATEGY_DIR = (
    ROOT
    / "quant/data_file/reports/strategy_agent_latest_l4_weight_candidates_20260702/postrank_open_filter_candidates/code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant/main/run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


def load_source() -> pd.DataFrame:
    sig = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(ACTIVE_WIDE), read_only=True)
    wide = con.execute(
        """
        select
            cast(signal_date as varchar) as signal_date,
            stock_code,
            buy_open_gap_raw_pct,
            ret_h1
        from active_l4_wide
        """
    ).fetchdf()
    con.close()
    wide["stock_code"] = wide["stock_code"].astype(str)
    df = sig.merge(wide, on=["signal_date", "stock_code"], how="left", suffixes=("", "_active"))
    if df["ret_h1"].isna().any():
        missing = int(df["ret_h1"].isna().sum())
        raise RuntimeError(f"active wide ret_h1 missing for {missing} source rows")
    return df


def max_drawdown(series: pd.Series) -> float:
    curve = (1.0 + series.fillna(0.0)).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def local_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"days": 0, "annual_proxy": np.nan, "sharpe_proxy": np.nan, "max_drawdown_proxy": np.nan}
    daily = df.groupby("buy_date", sort=True).apply(
        lambda x: float((x["ret_h1"] * x["target_pct_float"]).sum())
    )
    n = len(daily)
    curve_end = float((1.0 + daily).prod())
    annual = curve_end ** (252.0 / n) - 1.0 if n and curve_end > 0 else -1.0
    std = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / std * np.sqrt(252.0)) if std > 0 else np.nan
    return {
        "days": int(n),
        "annual_proxy": annual,
        "sharpe_proxy": sharpe,
        "max_drawdown_proxy": abs(max_drawdown(daily)),
        "win_day_ratio_proxy": float((daily > 0).mean()) if n else np.nan,
    }


def cases() -> list[dict]:
    out = []
    for gap_max in [-0.5, 0.0, 0.5, 1.0]:
        for pct_max in [-2.0, -2.5, -3.0, -4.0]:
            for target in [0.45, 0.48, 0.50, 0.53, 0.56]:
                out.append(
                    {
                        "name": f"filter_gap{gap_max}_pct{pct_max}_t{target}",
                        "mode": "filter",
                        "gap_max": gap_max,
                        "pct_max": pct_max,
                        "target": target,
                    }
                )
    for base, strong, weak in [
        (0.43, 0.52, 0.28),
        (0.43, 0.55, 0.25),
        (0.435, 0.54, 0.30),
        (0.44, 0.56, 0.25),
        (0.45, 0.60, 0.22),
    ]:
        out.append(
            {
                "name": f"dyn_base{base}_strong{strong}_weak{weak}",
                "mode": "dynamic",
                "base": base,
                "strong": strong,
                "weak": weak,
            }
        )
    return out


def apply_case(src: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = src.copy()
    if case["mode"] == "filter":
        df = df[(df["buy_open_gap_raw_pct"] <= case["gap_max"]) & (df["pct_chg"] <= case["pct_max"])].copy()
        df["target_pct_float"] = float(case["target"])
    elif case["mode"] == "dynamic":
        base = float(case["base"])
        strong = float(case["strong"])
        weak = float(case["weak"])
        cond_strong = (df["buy_open_gap_raw_pct"] <= -0.75) | (df["pct_chg"] <= -5.0)
        cond_weak = (df["buy_open_gap_raw_pct"] > 0.5) | (df["pct_chg"] > -2.35) | (df["pred_10d"] > 0.9985)
        df["target_pct_float"] = base
        df.loc[cond_strong, "target_pct_float"] = strong
        df.loc[cond_weak, "target_pct_float"] = weak
    else:
        raise ValueError(case["mode"])
    df = df.sort_values(["signal_date", "rank", "stock_code"]).copy()
    df["target_pct"] = df["target_pct_float"].map(lambda x: f"{x:.5f}")
    df["score_exit_entry_ratio"] = "0.96000"
    df["signal_stop_loss_pct"] = "0.05000"
    df["signal_take_profit_pct"] = "0.08000"
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
    return df


def write_signal(df: pd.DataFrame, name: str) -> Path:
    case_dir = SIGNAL_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "signals.csv"
    drop_cols = ["ret_h1", "target_pct_float"]
    if "buy_open_gap_raw_pct" in df.columns:
        # Keep explicit raw field for audit while preserving legacy columns used by runner.
        pass
    keep = [c for c in df.columns if c not in drop_cols]
    df[keep].to_csv(out, index=False, encoding="utf-8")
    return out


def extract_indicator(stdout: str) -> dict:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        try:
            return ast.literal_eval(indicator)
        except Exception:
            return {}
    return indicator or {}


def run_case(row: pd.Series) -> dict:
    signal_file = Path(row["signal_file"])
    name = row["name"]
    log_file = LOG_DIR / f"{name}.log"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        "3",
        "--holding-days",
        "1",
        "--max-holding-days",
        "1",
        "--score-exit-entry-ratio",
        "0.96",
        "--min-holding-days-before-score-exit",
        "1",
        "--score-continue-entry-ratio",
        "9.99",
        "--light-stop-loss-pct",
        "0.05",
        "--min-holding-days-before-light-stop",
        "1",
        "--take-profit-pct",
        "0.08",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (LOG_DIR / f"{name}.runner.json").write_text(proc.stdout, encoding="utf-8")
    indicator = extract_indicator(proc.stdout) if proc.returncode == 0 else {}
    return {
        **row.to_dict(),
        "returncode": proc.returncode,
        "log_file": str(log_file),
        "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
        "sharp_ratio": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "win_ratio": indicator.get("win_ratio"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    src = load_source()
    rows = []
    for case in cases():
        sig = apply_case(src, case)
        if sig["signal_date"].nunique() < 80:
            continue
        signal_file = write_signal(sig, case["name"])
        stats = local_stats(sig)
        rows.append(
            {
                **case,
                "signal_rows": int(len(sig)),
                "signal_days": int(sig["signal_date"].nunique()),
                "avg_target": float(sig["target_pct"].astype(float).mean()),
                "max_target": float(sig["target_pct"].astype(float).max()),
                "signal_file": str(signal_file),
                **stats,
            }
        )
    local = pd.DataFrame(rows)
    local = local.sort_values(["sharpe_proxy", "annual_proxy"], ascending=False).reset_index(drop=True)
    local.to_csv(LOCAL_SUMMARY, index=False, encoding="utf-8-sig")
    print("LOCAL_TOP")
    print(local.head(15).to_string(index=False))

    # Run a small set: top local Sharpe plus a few high annual candidates.
    pick = pd.concat(
        [
            local.head(8),
            local.sort_values(["annual_proxy", "sharpe_proxy"], ascending=False).head(4),
        ]
    ).drop_duplicates("name")
    results = []
    for _, row in pick.iterrows():
        print("RUN", row["name"], flush=True)
        results.append(run_case(row))
    out = pd.DataFrame(results).sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=False)
    out.to_csv(JUEJIN_SUMMARY, index=False, encoding="utf-8-sig")
    print("JUEJIN")
    print(
        out[
            [
                "name",
                "signal_days",
                "annual_proxy",
                "sharpe_proxy",
                "pnl_ratio_annual",
                "sharp_ratio",
                "max_drawdown",
                "open_count",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
