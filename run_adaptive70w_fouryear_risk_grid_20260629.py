from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_adaptive70w_fouryear_grid_20260629.py"


ENTRIES = [
    {"name": "w70_25_05_amt90_mv20", "w10d": 0.70, "w5d": 0.25, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w72_23_05_amt90_mv20", "w10d": 0.72, "w5d": 0.23, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w75_20_05_amt90_mv20", "w10d": 0.75, "w5d": 0.20, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
]

EXECS = [
    {
        "name": "hold4m6_c096_e094_ddxtight_pos55",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.55,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.04,
        "dd_hard": 0.08,
        "dd_soft_scale": 0.50,
        "dd_hard_scale": 0.30,
    },
    {
        "name": "hold4m6_c096_e094_ddxtight_pos60",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.60,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.04,
        "dd_hard": 0.08,
        "dd_soft_scale": 0.50,
        "dd_hard_scale": 0.30,
    },
    {
        "name": "hold4m6_c096_e094_ddxtight_pos65",
        "holding_days": 4,
        "max_holding_days": 6,
        "continue_ratio": 0.96,
        "exit_ratio": 0.94,
        "target_pct": 0.65,
        "stop_loss": 0.05,
        "take_profit": 0.09,
        "dd_soft": 0.04,
        "dd_hard": 0.08,
        "dd_soft_scale": 0.50,
        "dd_hard_scale": 0.30,
    },
]


def _load_base_module():
    spec = importlib.util.spec_from_file_location("fouryear_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = _load_base_module()
    tune_mod = mod._load_tune_module()
    manifest, _, sources = tune_mod._load_strategy(tune_mod.STRATEGY_DIR)
    detail_rows = []
    summary_rows = []
    for entry in ENTRIES:
        for exe in EXECS:
            case = tune_mod._build_case_assets(manifest, sources, entry, exe)
            rows = [mod._run(case, exe, tag, start, end) for tag, start, end in mod.SLICES]
            detail_rows.extend(rows)
            summary_rows.append(mod._summary(entry, exe, case, rows))
            mod._write_rows(mod.REPORT_DIR / "adaptive70w_fouryear_risk_detail.csv", detail_rows)
            mod._write_rows(mod.REPORT_DIR / "adaptive70w_fouryear_risk_summary.csv", summary_rows)
            print(json.dumps(summary_rows[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
