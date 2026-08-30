from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "merged_signal_pool_cache.duckdb"
OUT_DIR = REPORT_DIR / "pool_enriched_small_variants"


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _metric(sel: pd.DataFrame, hold: int, topn: int, target: float) -> dict | None:
    if sel.empty:
        return None
    ret_col = f"ret_h{hold}"
    picked = (
        sel.sort_values(["signal_date", "select_score", "stock_code"], ascending=[True, False, True])
        .groupby("signal_date", group_keys=False)
        .head(topn)
        .copy()
    )
    if len(picked) < 60:
        return None
    picked["_ret"] = (1.0 + picked[ret_col]) * (1.0 - 0.0015) / (1.0 + 0.0015) - 1.0
    daily = []
    wins = losses = 0
    for _, g in picked.groupby("signal_date"):
        scale = min(1.0, 1.0 / (target * len(g)))
        daily.append(float((g["_ret"] * target * scale).sum()))
        wins += int((g["_ret"] > 0).sum())
        losses += int((g["_ret"] < 0).sum())
    if len(daily) < 40:
        return None
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in daily:
        eq *= max(0.0001, 1.0 + r)
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
    years = len(daily) / 252.0
    annual = eq ** (1.0 / years) - 1.0
    mean = sum(daily) / len(daily)
    var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1) if len(daily) > 1 else 0.0
    sharpe = mean / math.sqrt(var) * math.sqrt(252.0) if var > 0 else float("nan")
    return {
        "local_annual": annual,
        "local_sharpe": sharpe,
        "local_max_drawdown": mdd,
        "rows": int(len(picked)),
        "days": int(len(daily)),
        "win_ratio": wins / (wins + losses) if wins + losses else float("nan"),
    }


def _score(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "liq":
        return df["blend_score"] + 0.03 * df["amount_pctile"] + 0.02 * df["mv_pctile"]
    if mode == "turnlow":
        return df["blend_score"] - 0.04 * df["turnover_pctile"] + 0.02 * df["amount_pctile"]
    if mode == "buyliq":
        return df["blend_score"] + 0.03 * df["buy_amount_pctile"] - 0.02 * df["buy_turnover_pctile"]
    return df["blend_score"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    df = con.execute("SELECT * FROM pool_enriched_small").fetchdf()
    con.close()
    rows = []
    for pct_max in [-1.75, -3.0, -5.0, -7.0, -10.0]:
        for signal_gap_low, signal_gap_high in [(-8.0, 1.5), (-5.0, 1.5), (-3.0, 0.0), (-1.0, 0.0)]:
            for buy_gap_low, buy_gap_high in [(-0.08, 0.03), (-0.05, 0.015), (-0.03, 0.01), (-0.01, 0.01)]:
                for r10 in [0.70, 0.80, 0.90, 0.95]:
                    for r1 in [0.0, 0.60, 0.80, 0.90]:
                        base = df[
                            (df["pct_chg"] <= pct_max)
                            & (df["buy_open_gap_raw_pct"].between(signal_gap_low, signal_gap_high))
                            & (df["buy_day_open_gap"].between(buy_gap_low, buy_gap_high))
                            & (df["rank_10d"] >= r10)
                            & (df["rank_1d"] >= r1)
                        ].copy()
                        if len(base) < 80:
                            continue
                        for mode in ["blend", "liq", "turnlow", "buyliq"]:
                            base["select_score"] = _score(base, mode)
                            for topn in [1, 2, 3]:
                                target = min(0.99, 0.99 / topn)
                                for hold in [1, 2, 3, 5]:
                                    m = _metric(base, hold, topn, target)
                                    if not m:
                                        continue
                                    name = (
                                        f"pes_pct{pct_max}_sg{signal_gap_low}_{signal_gap_high}"
                                        f"_bg{buy_gap_low}_{buy_gap_high}_r10{r10}_r1{r1}_{mode}_top{topn}_h{hold}"
                                    )
                                    name = re.sub(r"[^A-Za-z0-9_]+", "_", name.replace("-", "m").replace(".", "p"))
                                    rows.append(
                                        {
                                            "name": name,
                                            "pct_max": pct_max,
                                            "signal_gap_low": signal_gap_low,
                                            "signal_gap_high": signal_gap_high,
                                            "buy_gap_low": buy_gap_low,
                                            "buy_gap_high": buy_gap_high,
                                            "r10": r10,
                                            "r1": r1,
                                            "mode": mode,
                                            "topn": topn,
                                            "hold": hold,
                                            "target": target,
                                            **m,
                                        }
                                    )
    summary = pd.DataFrame(rows).sort_values(
        ["local_annual", "local_sharpe", "local_max_drawdown"],
        ascending=[False, False, True],
    )
    summary_path = REPORT_DIR / "pool_enriched_small_local_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest = []
    for row in summary.head(12).to_dict("records"):
        base = df[
            (df["pct_chg"] <= row["pct_max"])
            & (df["buy_open_gap_raw_pct"].between(row["signal_gap_low"], row["signal_gap_high"]))
            & (df["buy_day_open_gap"].between(row["buy_gap_low"], row["buy_gap_high"]))
            & (df["rank_10d"] >= row["r10"])
            & (df["rank_1d"] >= row["r1"])
        ].copy()
        base["select_score"] = _score(base, row["mode"])
        sig = (
            base.sort_values(["signal_date", "select_score", "stock_code"], ascending=[True, False, True])
            .groupby("signal_date", group_keys=False)
            .head(int(row["topn"]))
            .copy()
        )
        sig["rank"] = sig.groupby("signal_date").cumcount() + 1
        sig["symbol"] = sig["stock_code"].map(_symbol)
        out = sig[
            [
                "signal_date",
                "buy_date",
                "symbol",
                "stock_code",
                "name",
                "rank",
                "entry_score",
                "pred_1d",
                "pred_3d",
                "pred_5d",
                "pred_10d",
                "rank_1d",
                "rank_3d",
                "rank_5d",
                "rank_10d",
                "amount",
                "turnover_rate",
                "total_mv",
                "atr_qfq",
                "pct_chg",
                "buy_open_gap_raw_pct",
            ]
        ].copy()
        out.insert(6, "pred_prob", out["entry_score"])
        out["target_pct"] = f"{row['target']:.5f}"
        out["holding_days"] = int(row["hold"])
        out["max_holding_days"] = int(row["hold"])
        out["score_exit_entry_ratio"] = "9.99000"
        out["min_holding_days_before_score_exit"] = 1
        out["score_continue_entry_ratio"] = "9.99000"
        out["signal_stop_loss_pct"] = ""
        out["signal_take_profit_pct"] = ""
        out["strategy_variant"] = row["name"]
        out["filter_name"] = "pool_enriched_small"
        out["entry_weight_name"] = row["mode"]
        out["dynamic_hold_name"] = f"h{int(row['hold'])}"
        out["buy_day_market_available"] = True
        out["buy_day_hard_gate_complete"] = True
        out["buy_day_st_rejected"] = False
        out["buy_day_open_limit_up_rejected"] = False
        out["latest_market_date"] = "20260702"
        out_file = OUT_DIR / f"{row['name']}.csv"
        out.to_csv(out_file, index=False, encoding="utf-8")
        manifest.append(
            {
                "name": row["name"],
                "signal_file": str(out_file),
                "local_annual": row["local_annual"],
                "local_sharpe": row["local_sharpe"],
                "local_max_drawdown": row["local_max_drawdown"],
                "rows": row["rows"],
                "days": row["days"],
                "hold": row["hold"],
                "topn": row["topn"],
                "target": row["target"],
            }
        )
    manifest_path = OUT_DIR / "pool_enriched_small_top12_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["name"])
        writer.writeheader()
        writer.writerows(manifest)
    (OUT_DIR / "pool_enriched_small_top12_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.head(20).to_string(index=False))
    print(summary_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
