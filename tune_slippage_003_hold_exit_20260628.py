from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tune_slippage_003_candidates_20260628 import (
    REPORT_DIR as BASE_REPORT_DIR,
    TIME_SLICES,
    _load_tune_module,
    _run_backtest,
    _summary_row,
    _write_rows,
)


REPORT_DIR = BASE_REPORT_DIR

ENTRY_NAMES = [
    "w80_15_05_amt90_mv20",
    "w75_20_05_amt90_mv20",
]

CUSTOM_EXECS = [
    {
        "name": "hold3m5_c096_e095_ddtight",
        "holding_days": 3,
        "max_holding_days": 5,
        "continue_ratio": 0.96,
        "exit_ratio": 0.95,
        "target_pct": 0.90,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "hold3m5_c097_e096_ddtight",
        "holding_days": 3,
        "max_holding_days": 5,
        "continue_ratio": 0.97,
        "exit_ratio": 0.96,
        "target_pct": 0.90,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "hold4m6_c096_e094_ddtight",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.90,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "hold3m5_c096_e095_ddtight_pos80",
        "holding_days": 3,
        "max_holding_days": 5,
        "continue_ratio": 0.96,
        "exit_ratio": 0.95,
        "target_pct": 0.80,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
]


def main() -> None:
    mod = _load_tune_module()
    manifest, _, sources = mod._load_strategy(mod.STRATEGY_DIR)
    entries = {entry["name"]: entry for entry in mod.ENTRY_CASES}
    detail_rows = []
    summary_rows = []
    for entry_name in ENTRY_NAMES:
        entry = entries[entry_name]
        for exe in CUSTOM_EXECS:
            case = mod._build_case_assets(manifest, sources, entry, exe)
            case_key = case["case_key"]
            rows = [_run_backtest(entry, exe, case_key, tag, start, end) for tag, start, end in TIME_SLICES]
            detail_rows.extend(rows)
            summary_rows.append(_summary_row(rows))
            _write_rows(REPORT_DIR / "hold_exit_detail.csv", detail_rows)
            _write_rows(REPORT_DIR / "hold_exit_summary.csv", summary_rows)
            print(json.dumps(summary_rows[-1], ensure_ascii=False), flush=True)
    (REPORT_DIR / "hold_exit_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
