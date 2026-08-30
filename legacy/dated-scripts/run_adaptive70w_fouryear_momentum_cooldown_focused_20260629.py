from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_adaptive70w_fouryear_momentum_cooldown_20260629.py"
OUT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_adaptive70w_fouryear_momentum_cooldown_focused_20260629"
)


RULES = [
    {
        "name": "cool2d15",
        "description": "skip if signal-day two-trading-day cumulative return >= 15%",
        "two_day_cap": 0.15,
        "combo_cap": None,
        "turnover_floor": None,
    },
    {
        "name": "cool2d18",
        "description": "skip if signal-day two-trading-day cumulative return >= 18%",
        "two_day_cap": 0.18,
        "combo_cap": None,
        "turnover_floor": None,
    },
    {
        "name": "cool2d20",
        "description": "skip if signal-day two-trading-day cumulative return >= 20%",
        "two_day_cap": 0.20,
        "combo_cap": None,
        "turnover_floor": None,
    },
    {
        "name": "cool2d25",
        "description": "skip if signal-day two-trading-day cumulative return >= 25%",
        "two_day_cap": 0.25,
        "combo_cap": None,
        "turnover_floor": None,
    },
    {
        "name": "cool2d15_turn15",
        "description": "skip if two-day return >= 15% and signal-day turnover >= 15%",
        "two_day_cap": None,
        "combo_cap": 0.15,
        "turnover_floor": 15.0,
    },
    {
        "name": "cool2d20_turn20",
        "description": "skip if two-day return >= 20% and signal-day turnover >= 20%",
        "two_day_cap": None,
        "combo_cap": 0.20,
        "turnover_floor": 20.0,
    },
]

TARGETS = [0.45, 0.50, 0.55, 0.60]
SELL_MULTS = ["0.25"]


def _load_base():
    spec = importlib.util.spec_from_file_location("momentum_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT_DIR = OUT_DIR
    mod.RULES = RULES
    mod.TARGETS = TARGETS
    return mod


def main() -> None:
    base = _load_base()
    detail: list[dict[str, Any]] = []
    for target in TARGETS:
        for rule in RULES:
            case_meta = base._ensure_signal(rule, target)
            for sell_mult in SELL_MULTS:
                for tag, start, end in base.SLICES:
                    detail.append(base._run(case_meta, target, sell_mult, tag, start, end))
                    base._write_rows(OUT_DIR / "focused_detail.csv", detail)

    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for row in detail:
        item = summary.setdefault(
            (row["case_key"], row["sell_mult"]),
            {"case_key": row["case_key"], "sell_mult": row["sell_mult"], "signal_rows": row["signal_rows"]},
        )
        for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count", "log_file"]:
            item[f"{row['slice']}_{key}"] = row.get(key)
    base._write_rows(OUT_DIR / "focused_summary.csv", list(summary.values()))
    print("wrote", OUT_DIR / "focused_summary.csv")


if __name__ == "__main__":
    main()
