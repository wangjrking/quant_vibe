from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import re
from pathlib import Path

import pandas as pd

import tune_top1_soft_rank_latest_formal_20260623 as base_mod


ROOT = Path(r"D:\work\quant\quant_mcp")
OUT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_low_path_admission_20260623"
)

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("near_20250626", "2025-06-26 09:00:00"),
    ("near_20250701", "2025-07-01 09:00:00"),
    ("near_20250704", "2025-07-04 09:00:00"),
    ("near_20250926", "2025-09-26 09:00:00"),
    ("near_20251009", "2025-10-09 09:00:00"),
    ("near_20251014", "2025-10-14 09:00:00"),
    ("near_20251229", "2025-12-29 09:00:00"),
    ("near_20260105", "2026-01-05 09:00:00"),
    ("near_20260108", "2026-01-08 09:00:00"),
]

CANDIDATES = []
for weight_10d in [0.70, 0.75, 0.80, 0.83, 0.84, 0.85, 0.86, 0.87]:
    for holding_days in [4, 5]:
        for gamma, exit_ratio in [(1.8, 0.97), (2.0, 0.98)]:
            weight_5d = round(1.0 - weight_10d, 2)
            CANDIDATES.append(
                {
                    "name": f"lp_w{int(weight_10d * 100)}_5d{int(weight_5d * 100)}_h{holding_days}_g{str(gamma).replace('.', 'p')}_e{int(exit_ratio * 100)}",
                    "weights": {"10d": weight_10d, "5d": weight_5d},
                    "holding_days": holding_days,
                    "gamma": gamma,
                    "exit_ratio": exit_ratio,
                    "min_hold": 1,
                }
            )


def _write_rows(path: Path, rows: list[dict]) -> None:
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


def _summarize(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for candidate in sorted({row["candidate"] for row in rows}):
        items = [row for row in rows if row["candidate"] == candidate]
        full = next(row for row in items if row["start_name"] == "full_20240605")
        annuals = [float(row["annual"]) for row in items if row["annual"] is not None]
        near = [float(row["annual"]) for row in items if row["start_name"].startswith("near_") and row["annual"] is not None]
        if full["annual"] is None or not near:
            continue
        maxdd = max(float(row["max_drawdown"]) for row in items if row["max_drawdown"] is not None)
        full_annual = float(full["annual"])
        near_min = min(near)
        near_median = float(pd.Series(near).median())
        near_mean = float(pd.Series(near).mean())
        # Admission-oriented score: punish start-date weakness and drawdown more than full-period peak.
        score = full_annual * 0.25 + near_min * 0.30 + near_median * 0.30 + near_mean * 0.15 - max(0.0, maxdd - 0.35) * 1.5
        summary.append(
            {
                "candidate": candidate,
                "weights_json": full["weights_json"],
                "holding_days": full["holding_days"],
                "gamma": full["gamma"],
                "exit_ratio": full["exit_ratio"],
                "min_hold": full["min_hold"],
                "full_annual": full_annual,
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "near_min_annual": near_min,
                "near_median_annual": near_median,
                "near_mean_annual": near_mean,
                "near_max_annual": max(near),
                "worst_max_drawdown": maxdd,
                "annual_range": max(annuals) - min(annuals),
                "score": score,
            }
        )
    summary.sort(key=lambda row: (row["score"], row["near_min_annual"], row["full_annual"]), reverse=True)
    return summary


def _rows_from_logs() -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"(.+)_(full_20240605|near_\d{8})\.log$")
    for log_file in (OUT_DIR / "logs").glob("*.log"):
        match = pattern.match(log_file.name)
        if not match:
            continue
        candidate, start_name = match.groups()
        indicator = None
        for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if "GM_BACKTEST_INDICATOR:" not in line:
                continue
            payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
            try:
                indicator = ast.literal_eval(payload)
            except Exception:
                indicator = eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
            break
        if not indicator:
            continue
        rows.append(
            {
                "candidate": candidate,
                "start_name": start_name,
                "annual": indicator.get("pnl_ratio_annual"),
                "pnl_ratio": indicator.get("pnl_ratio"),
                "sharpe": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "win_ratio": indicator.get("win_ratio"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "log_file": str(log_file),
            }
        )
    return sorted(rows, key=lambda row: (row["candidate"], row["start_name"]))


def main() -> int:
    base_mod.OUT_DIR = OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = base_mod._build_base()
    market = base_mod._market_rows()
    next_date = base_mod._date_map(base)

    rows: list[dict] = _rows_from_logs()
    stats: list[dict] = []
    total = len(CANDIDATES) * len(STARTS)
    done = 0
    for cfg in CANDIDATES:
        signal_file = base_mod._write_signal(base, market, next_date, cfg)
        stats.append({"candidate": cfg["name"], **base_mod._signal_stats(signal_file)})
        score_db, score_table = base_mod._write_score_table(base, cfg)
        for start_name, start in STARTS:
            done += 1
            row = base_mod._run_case(cfg, signal_file, score_db, score_table, start_name, start)
            rows = [
                old
                for old in rows
                if not (old["candidate"] == row["candidate"] and old["start_name"] == row["start_name"])
            ]
            rows.append(row)
            rows.sort(key=lambda item: (item["candidate"], item["start_name"]))
            print(
                f"[{done}/{total}] {cfg['name']} {start_name} annual={row['annual']} "
                f"sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
            _write_rows(OUT_DIR / "low_path_cases.csv", rows)
    _write_rows(OUT_DIR / "low_path_signal_stats.csv", stats)
    summary = _summarize(rows)
    _write_rows(OUT_DIR / "low_path_summary.csv", summary)
    (OUT_DIR / "low_path_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
