from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
TUNE_MODULE_PATH = MAIN / "tune_current_prod_four_year_formal_20260628.py"


ENTRIES = [
    {"name": "w75_20_05_amt90_mv20", "w10d": 0.75, "w5d": 0.20, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w76_19_05_amt90_mv20", "w10d": 0.76, "w5d": 0.19, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w78_17_05_amt90_mv20", "w10d": 0.78, "w5d": 0.17, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w80_15_05_amt90_mv20", "w10d": 0.80, "w5d": 0.15, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
]

EXECS = [
    {
        "name": "hold4m6_c096_e094_ddtight_pos70",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.70,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "hold4m6_c096_e094_ddtight_pos75",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.75,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
]


def _load_tune_module():
    spec = importlib.util.spec_from_file_location("tune_mod", TUNE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TUNE_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = _load_tune_module()
    manifest, _, sources = mod._load_strategy(mod.STRATEGY_DIR)
    for entry in ENTRIES:
        for exe in EXECS:
            case = mod._build_case_assets(manifest, sources, entry, exe)
            print(f"{case['case_key']},{case['signal_rows']},{case['signal_file']},{case['score_table']}", flush=True)


if __name__ == "__main__":
    main()
