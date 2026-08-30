from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


NAV_RE = re.compile(r"GM_DAILY_NAV\s+(\d{8})\s+([0-9.]+)")


def period_metrics(
    nav: pd.Series, start: str, end: str
) -> dict[str, float | int]:
    prior = nav[nav.index < start]
    current = nav[(nav.index >= start) & (nav.index <= end)]
    if prior.empty or current.empty:
        raise RuntimeError(f"净值区间不完整: {start}-{end}")
    series = pd.concat([prior.tail(1), current])
    daily = series.pct_change().dropna()
    cumulative = float(series.iloc[-1] / series.iloc[0] - 1.0)
    volatility = float(daily.std(ddof=1))
    sharpe = (
        float(math.sqrt(252.0) * daily.mean() / volatility)
        if volatility > 0
        else float("nan")
    )
    drawdown = series / series.cummax() - 1.0
    return {
        "trading_days": int(len(daily)),
        "cumulative_return": cumulative,
        "sharpe": sharpe,
        "max_drawdown": float(-drawdown.min()),
        "positive_day_ratio": float((daily > 0).mean()),
    }


def parse_log(path: Path) -> pd.Series:
    rows = []
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    for line in raw.decode(encoding, errors="replace").splitlines():
        match = NAV_RE.search(line)
        if match:
            rows.append((match.group(1), float(match.group(2))))
    if not rows:
        raise RuntimeError(f"日志没有GM_DAILY_NAV: {path}")
    frame = pd.DataFrame(rows, columns=["trade_date", "nav"])
    frame = frame.drop_duplicates("trade_date", keep="last").sort_values(
        "trade_date"
    )
    return frame.set_index("trade_date")["nav"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    detail = {}
    periods = {
        "development_2026_01_04": ("20260105", "20260430"),
        "validation_2026_05_06": ("20260506", "20260630"),
        "stress_2026_07": ("20260701", "20260729"),
        "ytd_2026": ("20260105", "20260729"),
    }
    for path in sorted(args.log_dir.glob("*.log")):
        nav = parse_log(path)
        case_id = path.stem
        case = {"case_id": case_id}
        detail[case_id] = {}
        for prefix, (start, end) in periods.items():
            metrics = period_metrics(nav, start, end)
            detail[case_id][prefix] = metrics
            for key, value in metrics.items():
                case[f"{prefix}_{key}"] = value
        rows.append(case)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        args.output_dir / "juejin_2026_period_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "juejin_2026_period_metrics.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(frame.to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
