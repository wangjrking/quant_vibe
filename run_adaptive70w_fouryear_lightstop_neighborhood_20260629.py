from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
BASE_GRID = MAIN / "run_adaptive70w_fouryear_grid_20260629.py"
BASE_LIGHT = MAIN / "run_adaptive70w_fouryear_lightstop_20260629.py"


ENTRIES = [
    {"name": "w70_25_05_amt90_mv20", "w10d": 0.70, "w5d": 0.25, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w72_23_05_amt90_mv20", "w10d": 0.72, "w5d": 0.23, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w75_20_05_amt90_mv20", "w10d": 0.75, "w5d": 0.20, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
]

EXECS = []
for pos in (0.60, 0.65, 0.70, 0.75):
    EXECS.append(
        {
            "name": f"hold4m6_c096_e094_ddtight_pos{int(pos * 100)}",
            "holding_days": 4,
            "max_holding_days": 6,
            "continue_ratio": 0.96,
            "exit_ratio": 0.94,
            "target_pct": pos,
            "stop_loss": 0.05,
            "take_profit": 0.09,
            "dd_soft": 0.06,
            "dd_hard": 0.10,
            "dd_soft_scale": 0.65,
            "dd_hard_scale": 0.45,
        }
    )

LIGHT_STOPS = [0.06, 0.07, 0.08, 0.09]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    grid = _load_module(BASE_GRID, "fouryear_grid")
    light = _load_module(BASE_LIGHT, "fouryear_light")
    tune_mod = grid._load_tune_module()
    manifest, _, sources = tune_mod._load_strategy(tune_mod.STRATEGY_DIR)
    detail_rows = []
    summary_rows = []
    for entry in ENTRIES:
        for exe in EXECS:
            case = tune_mod._build_case_assets(manifest, sources, entry, exe)
            case_for_light = {
                "case_key": case["case_key"],
                "holding_days": exe["holding_days"],
                "max_holding_days": exe["max_holding_days"],
                "target_pct": exe["target_pct"],
                "stop_loss": exe["stop_loss"],
                "take_profit": exe["take_profit"],
                "score_continue": exe["continue_ratio"],
                "score_exit": exe["exit_ratio"],
            }
            for light_stop in LIGHT_STOPS:
                rows = [light._run(case_for_light, light_stop, tag, start, end) for tag, start, end in light.SLICES]
                detail_rows.extend(rows)
                by_slice = {row["slice"]: row for row in rows}
                out = {
                    "variant": rows[0]["variant"],
                    "case_key": case["case_key"],
                    "entry": entry["name"],
                    "target_pct": exe["target_pct"],
                    "light_stop": light_stop,
                    "signal_rows": case["signal_rows"],
                }
                for slice_name in ["full", "recent120", "recent60", "ytd2026"]:
                    row = by_slice[slice_name]
                    for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count"]:
                        out[f"{slice_name}_{key}"] = row.get(key)
                summary_rows.append(out)
                print(json.dumps(out, ensure_ascii=False), flush=True)
                grid._write_rows(grid.REPORT_DIR / "adaptive70w_fouryear_lightstop_neighborhood_detail.csv", detail_rows)
                grid._write_rows(grid.REPORT_DIR / "adaptive70w_fouryear_lightstop_neighborhood_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
