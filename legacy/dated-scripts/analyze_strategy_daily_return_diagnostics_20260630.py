from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
OUT_DAILY = REPORT_DIR / "daily_return_diagnostics_20260630.csv"
OUT_BUCKET = REPORT_DIR / "daily_return_bucket_summary_20260630.csv"
OUT_JSON = REPORT_DIR / "daily_return_diagnostics_summary_20260630.json"


CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "top3_cool2d18_daydrop99_high_annual",
        "annual": 8.711277032601416,
        "sharpe": 1.2490317274844058,
        "max_drawdown": 0.27734158618328963,
        "signal_file": REPORT_DIR / "signals" / "w72_23_05_amt150_mv30_top3_pos25_cool2d18_h2m3_e097_c098_daydrop99.csv",
        "log_file": REPORT_DIR / "targeted_logs" / "w72_23_05_amt150_mv30_top3_pos25_cool2d18_h2m3_e097_c098_daydrop99__intraday_ddloose_maxsell1_sell0p0.log",
    },
    {
        "name": "top5_cool2d10_high_annual_sharpe",
        "annual": 8.554910874122525,
        "sharpe": 1.8176015882718721,
        "max_drawdown": 0.20197332389827574,
        "signal_file": REPORT_DIR / "signals" / "w72_23_05_amt150_mv30_top5_pos15_cool2d10_h2m3_e097_c098_daydrop99.csv",
        "log_file": REPORT_DIR / "agreement_filter_logs" / "w72_23_05_amt150_mv30_top5_pos15_cool2d10_h2m3_e097_c098_daydrop99__maxsell1_sell0p0.log",
    },
    {
        "name": "dyn_mild_09_12_15_high_sharpe",
        "annual": 5.731332542989966,
        "sharpe": 1.8663811890630113,
        "max_drawdown": 0.16397974934864862,
        "signal_file": REPORT_DIR / "dynamic_target_signals" / "dyn_mild_09_12_15.csv",
        "log_file": REPORT_DIR / "dynamic_target_logs" / "dyn_mild_09_12_15.log",
    },
    {
        "name": "dyn_agree_040_065_best_sharpe",
        "annual": 3.6738119474265267,
        "sharpe": 1.8915890214917694,
        "max_drawdown": 0.15852307131064144,
        "signal_file": REPORT_DIR / "dynamic_target_signals" / "dyn_agree_040_065.csv",
        "log_file": REPORT_DIR / "dynamic_target_logs" / "dyn_agree_040_065.log",
    },
]


EXPOSURE_RE = re.compile(
    r"EXPOSURE (?P<date>\d{8}) post_buy invested_pct=(?P<invested>[0-9.]+) "
    r"active_positions=(?P<positions>\d+) market_value=(?P<market_value>[0-9.]+) "
    r"nav=(?P<nav>[0-9.]+) cash=(?P<cash>[0-9.]+)"
)


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _load_nav(log_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EXPOSURE_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "date": match.group("date"),
                "invested_pct": float(match.group("invested")),
                "active_positions": int(match.group("positions")),
                "market_value": float(match.group("market_value")),
                "nav": float(match.group("nav")),
                "cash": float(match.group("cash")),
            }
        )
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        if idx == 0:
            item["daily_return"] = None
        else:
            item["daily_return"] = row["nav"] / rows[idx - 1]["nav"] - 1.0
        out.append(item)
    return out


def _load_signal_stats(signal_file: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            buy_date = str(row.get("buy_date") or "").strip()
            if buy_date:
                grouped[buy_date].append(row)

    stats: dict[str, dict[str, Any]] = {}
    numeric_fields = [
        "target_pct",
        "pred_prob",
        "entry_score",
        "rank_1d",
        "rank_3d",
        "rank_5d",
        "rank_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "pct_chg",
        "prev_pct_chg",
        "two_day_ret",
        "list_age_days",
    ]
    for date, rows in grouped.items():
        item: dict[str, Any] = {"signal_count": len(rows)}
        for field in numeric_fields:
            values = [_float(row.get(field)) for row in rows]
            values = [value for value in values if value is not None]
            item[f"avg_{field}"] = _mean(values)
            item[f"min_{field}"] = min(values) if values else None
            item[f"max_{field}"] = max(values) if values else None
        target_values = [_float(row.get("target_pct")) or 0.0 for row in rows]
        item["target_sum"] = sum(target_values)
        item["names"] = ",".join(row.get("name", "") for row in rows[:5])
        stats[date] = item
    return stats


def _bin_two_day(value: float | None) -> str:
    if value is None:
        return "missing"
    if value <= -0.10:
        return "<=-10%"
    if value <= -0.08:
        return "-10%~-8%"
    if value <= -0.06:
        return "-8%~-6%"
    if value <= -0.03:
        return "-6%~-3%"
    if value <= 0:
        return "-3%~0"
    if value <= 0.05:
        return "0~5%"
    return ">5%"


def _bin_pct(value: float | None) -> str:
    if value is None:
        return "missing"
    if value <= -8:
        return "<=-8%"
    if value <= -6:
        return "-8%~-6%"
    if value <= -4:
        return "-6%~-4%"
    if value <= -2:
        return "-4%~-2%"
    if value <= 0:
        return "-2%~0"
    if value <= 4:
        return "0~4%"
    return ">4%"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _summary(name: str, daily_rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [row["daily_return"] for row in daily_rows if row.get("daily_return") is not None]
    losses_2 = [row for row in daily_rows if row.get("daily_return") is not None and row["daily_return"] <= -0.02]
    losses_3 = [row for row in daily_rows if row.get("daily_return") is not None and row["daily_return"] <= -0.03]
    gains_3 = [row for row in daily_rows if row.get("daily_return") is not None and row["daily_return"] >= 0.03]

    def feature_mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        return _mean(values)

    return {
        "name": name,
        "return_days": len(returns),
        "mean_daily_return": _mean(returns),
        "std_daily_return": statistics.pstdev(returns) if len(returns) > 1 else None,
        "calc_sharpe": (_mean(returns) or 0.0) / statistics.pstdev(returns) * math.sqrt(252)
        if len(returns) > 1 and statistics.pstdev(returns) > 0
        else None,
        "q01": _quantile(returns, 0.01),
        "q05": _quantile(returns, 0.05),
        "q50": _quantile(returns, 0.50),
        "q95": _quantile(returns, 0.95),
        "loss_days_le_2pct": len(losses_2),
        "loss_days_le_3pct": len(losses_3),
        "gain_days_ge_3pct": len(gains_3),
        "loss3_avg_signal_two_day": feature_mean(losses_3, "avg_two_day_ret"),
        "loss3_avg_signal_pct": feature_mean(losses_3, "avg_pct_chg"),
        "loss3_avg_turnover": feature_mean(losses_3, "avg_turnover_rate"),
        "loss3_avg_target_sum": feature_mean(losses_3, "target_sum"),
        "gain3_avg_signal_two_day": feature_mean(gains_3, "avg_two_day_ret"),
        "gain3_avg_signal_pct": feature_mean(gains_3, "avg_pct_chg"),
        "gain3_avg_turnover": feature_mean(gains_3, "avg_turnover_rate"),
        "gain3_avg_target_sum": feature_mean(gains_3, "target_sum"),
        "worst_10": sorted(
            [
                {
                    "date": row["date"],
                    "daily_return": row["daily_return"],
                    "invested_pct": row.get("invested_pct"),
                    "active_positions": row.get("active_positions"),
                    "avg_pct_chg": row.get("avg_pct_chg"),
                    "avg_two_day_ret": row.get("avg_two_day_ret"),
                    "target_sum": row.get("target_sum"),
                    "names": row.get("names"),
                }
                for row in daily_rows
                if row.get("daily_return") is not None
            ],
            key=lambda row: row["daily_return"],
        )[:10],
    }


def main() -> None:
    all_daily: list[dict[str, Any]] = []
    all_buckets: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        nav_rows = _load_nav(candidate["log_file"])
        signal_stats = _load_signal_stats(candidate["signal_file"])
        daily_rows: list[dict[str, Any]] = []
        for row in nav_rows:
            stats = signal_stats.get(row["date"], {})
            item = {
                "candidate": candidate["name"],
                "source_annual": candidate["annual"],
                "source_sharpe": candidate["sharpe"],
                "source_max_drawdown": candidate["max_drawdown"],
                **row,
                **stats,
            }
            item["avg_two_day_bin"] = _bin_two_day(_float(item.get("avg_two_day_ret")))
            item["avg_pct_bin"] = _bin_pct(_float(item.get("avg_pct_chg")))
            daily_rows.append(item)
        all_daily.extend(daily_rows)
        summaries.append(_summary(candidate["name"], daily_rows))

        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in daily_rows:
            ret = row.get("daily_return")
            if ret is None:
                continue
            grouped[(row["avg_two_day_bin"], row["avg_pct_bin"])].append(ret)
        for (two_day_bin, pct_bin), returns in grouped.items():
            all_buckets.append(
                {
                    "candidate": candidate["name"],
                    "avg_two_day_bin": two_day_bin,
                    "avg_pct_bin": pct_bin,
                    "days": len(returns),
                    "mean_daily_return": _mean(returns),
                    "std_daily_return": statistics.pstdev(returns) if len(returns) > 1 else None,
                    "calc_sharpe": (_mean(returns) or 0.0) / statistics.pstdev(returns) * math.sqrt(252)
                    if len(returns) > 1 and statistics.pstdev(returns) > 0
                    else None,
                    "loss_days_le_2pct": sum(1 for value in returns if value <= -0.02),
                    "gain_days_ge_3pct": sum(1 for value in returns if value >= 0.03),
                    "q05": _quantile(returns, 0.05),
                    "q50": _quantile(returns, 0.50),
                    "q95": _quantile(returns, 0.95),
                }
            )

    _write_csv(OUT_DAILY, all_daily)
    _write_csv(OUT_BUCKET, all_buckets)
    OUT_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"daily": str(OUT_DAILY), "bucket": str(OUT_BUCKET), "summary": str(OUT_JSON)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
