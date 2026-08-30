from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_ROOT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "current_formal_top1keep_tailpow_continuous_vs_startup"
FULL_LOG = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "logs" / "tp_peak_g1p38_o0p942.log"
WARMUP_SUMMARY = REPORT_ROOT / "current_formal_top1keep_tailpow_warmup_state" / "warmup_state_summary.csv"

TARGETS = [
    ("2025-07-01", "20250701", "target_20250701"),
    ("2025-10-08", "20251008", "target_20251008"),
    ("2026-01-05", "20260105", "target_20260105"),
]

EXPOSURE_RE = re.compile(
    r"EXPOSURE\s+(?P<date>\d{8})\s+post_buy\s+"
    r"invested_pct=(?P<invested>[0-9.]+)\s+"
    r"active_positions=(?P<positions>\d+)\s+"
    r"market_value=(?P<market_value>[0-9.]+)\s+"
    r"nav=(?P<nav>[0-9.]+)\s+"
    r"cash=(?P<cash>[0-9.]+)"
)


def _parse_exposures(log_file: Path) -> list[dict]:
    rows: list[dict] = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
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


def _max_drawdown(nav_values: list[float]) -> float | None:
    if not nav_values:
        return None
    peak = nav_values[0]
    max_dd = 0.0
    for value in nav_values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - value / peak)
    return max_dd


def _slice_metrics(exposures: list[dict], start_date: str) -> dict:
    rows = [row for row in exposures if row["date"] >= start_date]
    if not rows:
        return {}
    start = rows[0]
    end = rows[-1]
    cumulative = end["nav"] / start["nav"] - 1.0
    return {
        "continuous_start_nav": start["nav"],
        "continuous_end_nav": end["nav"],
        "continuous_cumulative_return": cumulative,
        "continuous_annualized_return": _annualize(cumulative, len(rows)),
        "continuous_max_drawdown": _max_drawdown([row["nav"] for row in rows]),
        "continuous_trading_days": len(rows),
        "continuous_start_invested_pct": start["invested_pct"],
        "continuous_start_active_positions": start["active_positions"],
    }


def _load_warmup() -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    with WARMUP_SUMMARY.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows[(row["target"], int(row["warmup_signal_days"]))] = row
    return rows


def _float(row: dict, key: str) -> float | None:
    value = str(row.get(key) or "").strip()
    if value == "":
        return None
    return float(value)


def main() -> int:
    exposures = _parse_exposures(FULL_LOG)
    warmup = _load_warmup()
    output = []
    for display_date, yyyymmdd, target_tag in TARGETS:
        continuous = _slice_metrics(exposures, yyyymmdd)
        cold = warmup.get((target_tag, 0), {})
        warm60 = warmup.get((target_tag, 60), {})
        row = {
            "target_date": display_date,
            **continuous,
            "cold_start_annualized_return": _float(cold, "post_target_annualized_return"),
            "cold_start_cumulative_return": _float(cold, "post_target_cumulative_return"),
            "cold_start_max_drawdown": _float(cold, "post_target_max_drawdown"),
            "cold_start_active_positions": _float(cold, "target_start_active_positions"),
            "cold_start_invested_pct": _float(cold, "target_start_invested_pct"),
            "warmup60_annualized_return": _float(warm60, "post_target_annualized_return"),
            "warmup60_cumulative_return": _float(warm60, "post_target_cumulative_return"),
            "warmup60_max_drawdown": _float(warm60, "post_target_max_drawdown"),
            "warmup60_active_positions": _float(warm60, "target_start_active_positions"),
            "warmup60_invested_pct": _float(warm60, "target_start_invested_pct"),
        }
        output.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(output[0].keys())
    with (OUT_DIR / "continuous_vs_startup_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (OUT_DIR / "continuous_vs_startup_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for row in output:
        print(
            f"{row['target_date']} continuous annual={row['continuous_annualized_return']:.4f} "
            f"pnl={row['continuous_cumulative_return']:.4f} "
            f"cold annual={row['cold_start_annualized_return']:.4f} "
            f"warm60 annual={row['warmup60_annualized_return']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
