from __future__ import annotations

import json
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
OUT_DIR = REPORT_DIR / "repro500_feature_frontier"


def max_drawdown(daily: pd.Series) -> float:
    curve = (1.0 + daily.fillna(0.0)).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def daily_stats(df: pd.DataFrame, target_col: str = "target") -> dict:
    daily = df.groupby("buy_date", sort=True).apply(
        lambda x: float((x["ret_h1"] * x[target_col]).sum())
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
        "daily_mean": float(daily.mean()),
        "daily_std": std,
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
    }


def load_data() -> pd.DataFrame:
    sig = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(ACTIVE_WIDE), read_only=True)
    wide = con.execute(
        """
        select
            cast(signal_date as varchar) as signal_date,
            stock_code,
            buy_open_gap_raw_pct,
            signal_open_gap_raw_pct,
            signal_amount,
            signal_turnover_rate,
            signal_total_mv,
            signal_atr_qfq,
            ret_h1
        from active_l4_wide
        """
    ).fetchdf()
    con.close()
    wide["stock_code"] = wide["stock_code"].astype(str)
    df = sig.merge(wide, on=["signal_date", "stock_code"], how="left")
    if df["ret_h1"].isna().any():
        raise RuntimeError("missing active wide ret_h1")
    df["target"] = df["target_pct"].astype(float)
    return df


def candidate_masks(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    specs: list[tuple[str, pd.Series]] = []
    fields = {
        "pct": df["pct_chg"].astype(float),
        "bog": df["buy_open_gap_raw_pct"].astype(float),
        "pred1": df["pred_1d"].astype(float),
        "pred3": df["pred_3d"].astype(float),
        "pred10": df["pred_10d"].astype(float),
        "turn": df["turnover_rate"].astype(float),
        "amt": df["amount"].astype(float),
        "mv": df["total_mv"].astype(float),
        "atr": df["atr_qfq"].astype(float),
    }
    thresholds = {
        "pct": [-10, -8, -6, -5, -4, -3, -2.35, -1.75],
        "bog": [-6, -4, -2, -1, -0.5, 0, 0.5, 1.0],
        "pred1": [-0.002, -0.001, 0, 0.5, 0.9, 0.99, 0.9985],
        "pred3": [0, 0.5, 0.9, 0.99, 0.9985],
        "pred10": [0.7, 0.9, 0.97, 0.99, 0.9985],
        "turn": [0.5, 1, 2, 3, 5, 8, 12],
        "amt": [90000, 120000, 200000, 350000, 500000, 800000],
        "mv": [300000, 500000, 1000000, 3000000, 8000000],
        "atr": [0.25, 0.5, 1.0, 2.0, 3.0],
    }
    for name, series in fields.items():
        for th in thresholds[name]:
            specs.append((f"{name}_le_{th}", series <= th))
            specs.append((f"{name}_gt_{th}", series > th))
    # Common two-field combinations observed in previous attempts.
    specs.extend(
        [
            ("late_rebound", (fields["pct"] > -2.35) | (fields["bog"] > 0.5)),
            ("deep_drop_or_low_open", (fields["pct"] <= -5) | (fields["bog"] <= -1)),
            ("crowded", (fields["pred1"] > 0.9985) | (fields["pred10"] > 0.9985)),
            ("weak_late_or_crowded", ((fields["pct"] > -2.35) | (fields["bog"] > 0.5)) | ((fields["pred1"] > 0.9985) | (fields["pred10"] > 0.9985))),
            ("liquid_low_open", (fields["bog"] <= -1) & (fields["amt"] >= 200000)),
            ("deep_liquid", (fields["pct"] <= -5) & (fields["amt"] >= 200000)),
            ("high_atr_deep", (fields["atr"] >= 1.0) & (fields["pct"] <= -5)),
        ]
    )
    return specs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    base_stats = daily_stats(df)
    rows = []
    for name, mask in candidate_masks(df):
        mask = mask.fillna(False)
        count = int(mask.sum())
        if count < 8 or count > len(df) - 8:
            continue
        inside = df.loc[mask].copy()
        outside = df.loc[~mask].copy()
        rows.append(
            {
                "rule": name,
                "rows": count,
                "days": int(inside["buy_date"].nunique()),
                "inside_ret_mean": float(inside["ret_h1"].mean()),
                "inside_ret_std": float(inside["ret_h1"].std(ddof=1)),
                "inside_win": float((inside["ret_h1"] > 0).mean()),
                "outside_ret_mean": float(outside["ret_h1"].mean()),
                "outside_win": float((outside["ret_h1"] > 0).mean()),
                "mean_delta": float(inside["ret_h1"].mean() - outside["ret_h1"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["mean_delta", "inside_ret_mean"], ascending=[True, True])
    out.to_csv(OUT_DIR / "feature_subset_return_summary.csv", index=False, encoding="utf-8-sig")

    # Proxy target scans: reduce poor-looking masks and slightly raise non-mask rows.
    scans = []
    top_rules = out.head(35)["rule"].tolist()
    masks = dict(candidate_masks(df))
    for rule in top_rules:
        mask = masks[rule].fillna(False)
        for weak_target in [0.30, 0.34, 0.36, 0.38, 0.40]:
            for normal_target in [0.44, 0.45, 0.46, 0.47, 0.48]:
                tmp = df.copy()
                tmp["target_scan"] = normal_target
                tmp.loc[mask, "target_scan"] = weak_target
                stats = daily_stats(tmp, "target_scan")
                scans.append(
                    {
                        "rule": rule,
                        "weak_rows": int(mask.sum()),
                        "weak_target": weak_target,
                        "normal_target": normal_target,
                        **stats,
                    }
                )
    scan_df = pd.DataFrame(scans).sort_values(["sharpe_proxy", "annual_proxy"], ascending=False)
    scan_df.to_csv(OUT_DIR / "proxy_target_scan_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "base": base_stats,
        "best_bad_rules": out.head(20).to_dict(orient="records"),
        "best_proxy_scans": scan_df.head(30).to_dict(orient="records"),
    }
    (OUT_DIR / "feature_frontier_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
