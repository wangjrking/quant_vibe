from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_ROOT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "current_formal_top1keep_tailpow_continuous_anchor_sensitivity"
FULL_LOG = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "logs" / "tp_peak_g1p38_o0p942.log"

EXPOSURE_RE = re.compile(
    r"EXPOSURE\s+(?P<date>\d{8})\s+post_buy\s+"
    r"invested_pct=(?P<invested>[0-9.]+)\s+"
    r"active_positions=(?P<positions>\d+)\s+"
    r"market_value=(?P<market_value>[0-9.]+)\s+"
    r"nav=(?P<nav>[0-9.]+)\s+"
    r"cash=(?P<cash>[0-9.]+)"
)


def _parse_exposures() -> list[dict]:
    rows = []
    for line in FULL_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EXPOSURE_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "date": item["date"],
                "invested_pct": float(item["invested"]),
                "active_positions": int(item["positions"]),
                "nav": float(item["nav"]),
            }
        )
    return rows


def _annualize(cumulative_return: float, trading_days: int) -> float | None:
    if trading_days <= 0 or cumulative_return <= -1.0:
        return None
    return math.pow(1.0 + cumulative_return, 252.0 / trading_days) - 1.0


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - value / peak)
    return max_dd


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def main() -> int:
    exposures = _parse_exposures()
    rows = []
    for index, start in enumerate(exposures):
        sliced = exposures[index:]
        if len(sliced) < 20:
            continue
        cumulative = sliced[-1]["nav"] / start["nav"] - 1.0
        rows.append(
            {
                "start_date": start["date"],
                "end_date": sliced[-1]["date"],
                "trading_days": len(sliced),
                "start_nav": start["nav"],
                "end_nav": sliced[-1]["nav"],
                "cumulative_return": cumulative,
                "annualized_return": _annualize(cumulative, len(sliced)),
                "max_drawdown": _max_drawdown([row["nav"] for row in sliced]),
                "start_invested_pct": start["invested_pct"],
                "start_active_positions": start["active_positions"],
            }
        )

    annuals = [row["annualized_return"] for row in rows if row["annualized_return"] is not None]
    summary = {
        "source_log": str(FULL_LOG),
        "end_date": exposures[-1]["date"] if exposures else None,
        "anchor_count": len(rows),
        "annual_min": min(annuals),
        "annual_p10": _percentile(annuals, 0.10),
        "annual_p25": _percentile(annuals, 0.25),
        "annual_median": _percentile(annuals, 0.50),
        "annual_p75": _percentile(annuals, 0.75),
        "annual_p90": _percentile(annuals, 0.90),
        "annual_max": max(annuals),
        "min_row": min(rows, key=lambda row: row["annualized_return"]),
        "max_row": max(rows, key=lambda row: row["annualized_return"]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "continuous_anchor_sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "continuous_anchor_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
