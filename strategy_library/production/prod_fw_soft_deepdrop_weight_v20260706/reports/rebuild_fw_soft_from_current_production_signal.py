from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_param_search_20260705"
SOURCE = ROOT / "quant/main/strategy_library/production/prod_deepdrop_smallcap_refill_v20260704/signals/full_history_deepdrop_smallcap_refill.csv"
OUT_DIR = REPORT_DIR / "production_publish_candidates"
OUT_SIGNAL = OUT_DIR / "fw_soft_all_weak85_strong105_from_current_production_full_history.csv"
OUT_AUDIT = OUT_DIR / "fw_soft_all_weak85_strong105_from_current_production_full_history_audit.json"


def cap_daily_exposure(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    out = df.copy()
    daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
    scale = (cap / daily_sum).clip(upper=1.0)
    out["target_pct"] = out["target_pct"] * scale
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in ("signal_date", "buy_date"):
        df[col] = df[col].astype(str).str.replace(".0", "", regex=False).str.zfill(8)

    weak = (df["pct_chg"].astype(float) >= -2.5) | (df["buy_open_gap_raw_pct"].fillna(0).astype(float) >= 0)
    strong = (
        (df["pct_chg"].astype(float) <= -5.0)
        & (df["buy_open_gap_raw_pct"].fillna(0).astype(float) <= -0.8)
        & (df["turnover_rate"].astype(float) >= 4.5)
    )
    out = df.copy()
    out["feature_weight_scale"] = 1.0
    out.loc[weak, "feature_weight_scale"] = 0.85
    out.loc[strong, "feature_weight_scale"] = 1.05
    out["target_pct"] = out["target_pct"].astype(float) * out["feature_weight_scale"]
    out = cap_daily_exposure(out, 0.91)
    out["strategy_variant"] = "prod_fw_soft_deepdrop_weight_v20260706"
    out["source_strategy_variant"] = "fw_soft_rebuilt_from_current_production_full_history"
    out.to_csv(OUT_SIGNAL, index=False, encoding="utf-8-sig")

    latest_buy = out["buy_date"].max()
    latest = out[out["buy_date"] == latest_buy].copy()
    audit = {
        "schema_version": 1,
        "source": str(SOURCE),
        "output": str(OUT_SIGNAL),
        "row_count": int(len(out)),
        "signal_days": int(out["signal_date"].nunique()),
        "buy_days": int(out["buy_date"].nunique()),
        "stock_count": int(out["stock_code"].nunique()),
        "signal_date_min": str(out["signal_date"].min()),
        "signal_date_max": str(out["signal_date"].max()),
        "buy_date_min": str(out["buy_date"].min()),
        "buy_date_max": str(out["buy_date"].max()),
        "duplicate_buy_stock_keys": int(out.duplicated(["buy_date", "stock_code"]).sum()),
        "weak_rows": int(weak.sum()),
        "strong_rows": int(strong.sum()),
        "max_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().max()),
        "mean_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().mean()),
        "latest_buy_date_rows": int(len(latest)),
        "latest_buy_date_stock_codes": latest["stock_code"].astype(str).tolist(),
        "latest_buy_date_target_pct_sum": float(latest["target_pct"].sum()) if len(latest) else 0.0,
    }
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"signal": str(OUT_SIGNAL), "audit": str(OUT_AUDIT), **audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
