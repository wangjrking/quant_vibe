from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant" / "main" / "validate_current_formal_l4_5d10d_top3_independent_sell_20260720.py"
SPEC = importlib.util.spec_from_file_location("validation_common", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_best_validation_20260720"
)
MODULE.OUT = OUT
MODULE.LOGS = OUT / "logs"
MODULE.STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_score_quantized_20260720"
    / "research_code_snapshot"
)
MODULE.SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_entry_exit_grid_20260720"
    / "signals"
    / "buygap_gt5_scale50.csv"
)
MODULE.SCORE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720"
    / "score_assets"
    / "w10_100.duckdb"
)

END = "2026-07-17 15:30:00"
WINDOWS = [
    ("full_repeat_a", "2022-06-07 09:00:00", END),
    ("full_repeat_b", "2022-06-07 09:00:00", END),
    ("2022h2", "2022-06-07 09:00:00", "2022-12-30 15:30:00"),
    ("2023", "2023-01-03 09:00:00", "2023-12-29 15:30:00"),
    ("2024", "2024-01-02 09:00:00", "2024-12-31 15:30:00"),
    ("2025", "2025-01-02 09:00:00", "2025-12-31 15:30:00"),
    ("2026ytd", "2026-01-05 09:00:00", END),
    ("recent60", "2026-04-22 09:00:00", END),
    ("cold_202407_a", "2024-06-27 09:00:00", END),
    ("cold_202407_b", "2024-07-01 09:00:00", END),
    ("cold_202407_c", "2024-07-03 09:00:00", END),
    ("cold_202507_a", "2025-06-27 09:00:00", END),
    ("cold_202507_b", "2025-07-01 09:00:00", END),
    ("cold_202507_c", "2025-07-03 09:00:00", END),
    ("cold_202601_a", "2025-12-30 09:00:00", END),
    ("cold_202601_b", "2026-01-05 09:00:00", END),
    ("cold_202601_c", "2026-01-07 09:00:00", END),
]


def main() -> None:
    MODULE.LOGS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for window in WINDOWS:
        result = MODULE.run_window(*window)
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / "juejin_results.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    full = [row for row in rows if row["name"].startswith("full_repeat")]
    repeatable = len(full) == 2 and all(
        full[0].get(key) == full[1].get(key)
        for key in ("annual_return", "sharpe", "max_drawdown", "open_count", "close_count")
    )
    cold = [row for row in rows if row["name"].startswith("cold_")]
    payload = {
        "status": "research_only_not_admitted",
        "candidate": "buygap_gt5_scale50_currentcoverage",
        "signal_file": str(MODULE.SIGNAL),
        "signal_sha256": MODULE.hashlib.sha256(MODULE.SIGNAL.read_bytes()).hexdigest(),
        "score_db": str(MODULE.SCORE_DB),
        "score_db_sha256": MODULE.hashlib.sha256(MODULE.SCORE_DB.read_bytes()).hexdigest(),
        "market_db": str(MODULE.MARKET_DB),
        "backtest_adjust": "none",
        "one_side_slippage_ratio": 0.003,
        "exact_repeatable": repeatable,
        "cold_start_min_annual": min((row.get("annual_return") for row in cold if row.get("annual_return") is not None), default=None),
        "cold_start_min_sharpe": min((row.get("sharpe") for row in cold if row.get("sharpe") is not None), default=None),
        "cold_start_max_drawdown": max((row.get("max_drawdown") for row in cold if row.get("max_drawdown") is not None), default=None),
        "results": rows,
    }
    (OUT / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
