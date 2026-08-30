from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import duckdb


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
SIGNAL_BASE = BASE / "signals"
OUT_DIR = BASE / "open_gap_band_variants"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

SOURCES = [
    "w35_15_00_50_pctm1p75_gapm0p08to0p0_r1070_r160",
    "w25_25_00_50_pctm1p75_gapm0p08to0p0_r1070_r160",
    "w15_35_00_50_pctm1p75_gapm0p08to0p0_r1070_r160",
    "w35_15_00_50_pctm2p5_gapm0p08to0p0_r1070_r160",
    "w25_25_00_50_pctm2p5_gapm0p08to0p0_r1070_r160",
]


def _annualized_return(rets: list[float]) -> float:
    if not rets:
        return float("nan")
    equity = 1.0
    for ret in rets:
        equity *= max(0.0001, 1.0 + ret)
    years = len(rets) / 252.0
    if years <= 0 or equity <= 0:
        return float("nan")
    return equity ** (1.0 / years) - 1.0


def _sharpe(rets: list[float]) -> float:
    if len(rets) < 2:
        return float("nan")
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    std = math.sqrt(var)
    return mean / std * math.sqrt(252.0) if std else float("nan")


def _mdd(rets: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    out = 0.0
    for ret in rets:
        equity *= max(0.0001, 1.0 + ret)
        peak = max(peak, equity)
        out = max(out, 1.0 - equity / peak)
    return out


def _load_returns() -> dict[tuple[str, str], float]:
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        rows = con.execute(
            """
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS sell_date
                FROM (SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA)
            )
            SELECT
                buy.trade_date AS buy_date,
                buy.stock_code,
                sell.open / NULLIF(buy.open, 0) - 1 AS open_to_next_open_ret
            FROM STOCK_DAILY_DATA buy
            JOIN cal ON cal.trade_date = buy.trade_date
            JOIN STOCK_DAILY_DATA sell
              ON sell.trade_date = cal.sell_date
             AND sell.stock_code = buy.stock_code
            WHERE buy.open IS NOT NULL AND sell.open IS NOT NULL
            """
        ).fetchall()
    finally:
        con.close()
    return {(str(d), str(c)): float(r) for d, c, r in rows if r is not None}


def _evaluate(rows: list[dict], returns: dict[tuple[str, str], float], target: float) -> dict:
    by_day: dict[str, list[dict]] = {}
    wins = 0
    losses = 0
    for row in rows:
        key = (str(row["buy_date"]), str(row["stock_code"]))
        if key not in returns:
            continue
        row = dict(row)
        row["_ret"] = returns[key]
        by_day.setdefault(str(row["signal_date"]), []).append(row)
        if row["_ret"] > 0:
            wins += 1
        elif row["_ret"] < 0:
            losses += 1
    daily: list[float] = []
    for day_rows in by_day.values():
        total_target = target * len(day_rows)
        scale = min(1.0, 1.0 / total_target) if total_target > 0 else 0.0
        daily.append(sum(target * scale * r["_ret"] for r in day_rows))
    return {
        "local_annual": _annualized_return(daily),
        "local_sharpe": _sharpe(daily),
        "local_max_drawdown": _mdd(daily),
        "local_signal_days": len(by_day),
        "local_open_count": sum(len(v) for v in by_day.values()),
        "local_win_ratio": wins / (wins + losses) if wins + losses else float("nan"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = _load_returns()
    gap_bands = [(-8, -5), (-5, -3), (-3, -1.5), (-1.5, -0.5), (-0.5, 0), (-8, -1.5), (-5, 0), (-3, 0)]
    pct_bands = [(-20, -7), (-20, -5), (-12, -5), (-10, -5), (-8, -5), (-12, -3.5), (-8, -3.5)]
    topns = [1, 2, 3]
    targets = [0.435, 0.60, 0.80, 0.99]
    min_rank_pairs = [(0.70, 0.60), (0.80, 0.80), (0.90, 0.80), (0.90, 0.90), (0.95, 0.80)]
    manifest: list[dict] = []
    for source in SOURCES:
        source_file = SIGNAL_BASE / f"{source}.csv"
        if not source_file.exists():
            continue
        rows = list(csv.DictReader(source_file.open("r", encoding="utf-8-sig", newline="")))
        for gap_low, gap_high in gap_bands:
            for pct_low, pct_high in pct_bands:
                for r10_min, r1_min in min_rank_pairs:
                    base_rows = [
                        row
                        for row in rows
                        if gap_low <= float(row.get("buy_open_gap_raw_pct") or 999) <= gap_high
                        and pct_low <= float(row.get("pct_chg") or 999) <= pct_high
                        and float(row.get("rank_10d") or 0) >= r10_min
                        and float(row.get("rank_1d") or 0) >= r1_min
                    ]
                    if not base_rows:
                        continue
                    grouped: dict[str, list[dict]] = {}
                    for row in base_rows:
                        grouped.setdefault(str(row["signal_date"]), []).append(row)
                    for topn in topns:
                        selected: list[dict] = []
                        for _, day_rows in sorted(grouped.items()):
                            day_rows = sorted(day_rows, key=lambda r: int(r["rank"]))[:topn]
                            for idx, row in enumerate(day_rows, 1):
                                nr = dict(row)
                                nr["rank"] = str(idx)
                                selected.append(nr)
                        if not (80 <= len(selected) <= 900):
                            continue
                        for target in targets:
                            out_rows = []
                            for row in selected:
                                nr = dict(row)
                                nr["target_pct"] = f"{target:.5f}"
                                nr["strategy_variant"] = "active_l4_open_gap_band"
                                nr["filter_name"] = (
                                    f"gap{gap_low}to{gap_high}_pct{pct_low}to{pct_high}"
                                    f"_r10{int(r10_min*100)}_r1{int(r1_min*100)}"
                                )
                                out_rows.append(nr)
                            metrics = _evaluate(out_rows, returns, target)
                            if metrics["local_open_count"] == 0:
                                continue
                            name = (
                                f"{source}_top{topn}_t{int(target*100)}"
                                f"_gap{str(gap_low).replace('-', 'm').replace('.', 'p')}to{str(gap_high).replace('-', 'm').replace('.', 'p')}"
                                f"_pct{str(pct_low).replace('-', 'm').replace('.', 'p')}to{str(pct_high).replace('-', 'm').replace('.', 'p')}"
                                f"_r10{int(r10_min*100)}_r1{int(r1_min*100)}"
                            )
                            out_file = OUT_DIR / f"{name}.csv"
                            with out_file.open("w", encoding="utf-8", newline="") as file:
                                writer = csv.DictWriter(file, fieldnames=list(out_rows[0].keys()))
                                writer.writeheader()
                                writer.writerows(out_rows)
                            manifest.append(
                                {
                                    "name": name,
                                    "source": source,
                                    "signal_file": str(out_file),
                                    "rows": len(out_rows),
                                    "days": len({r["signal_date"] for r in out_rows}),
                                    "topn": topn,
                                    "target": target,
                                    "gap_low": gap_low,
                                    "gap_high": gap_high,
                                    "pct_low": pct_low,
                                    "pct_high": pct_high,
                                    "r10_min": r10_min,
                                    "r1_min": r1_min,
                                    **metrics,
                                }
                            )
    manifest = sorted(
        manifest,
        key=lambda r: (
            r["local_annual"],
            r["local_sharpe"],
            -r["local_max_drawdown"],
            r["local_open_count"],
        ),
        reverse=True,
    )
    out_manifest = OUT_DIR / "open_gap_band_manifest.csv"
    with out_manifest.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = list(manifest[0].keys()) if manifest else ["name"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    (OUT_DIR / "open_gap_band_manifest.json").write_text(
        json.dumps(manifest[:100], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for row in manifest[:20]:
        print(row)
    print(out_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
