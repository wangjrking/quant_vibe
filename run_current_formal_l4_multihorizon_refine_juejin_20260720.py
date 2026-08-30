from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant" / "main" / "run_current_formal_l4_persistent_edge_juejin_20260720.py"
SPEC = importlib.util.spec_from_file_location("persistent_edge_runner", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_multihorizon_refine_juejin_20260720"
)
MODULE.REPORT_DIR = REPORT_DIR
MODULE.SIGNAL_DIR = REPORT_DIR / "signals"
MODULE.SCORE_DIR = REPORT_DIR / "score_assets"
MODULE.LOG_DIR = REPORT_DIR / "logs"

for floor in [0.60, 0.65, 0.70, 0.75]:
    for pct_cap in [2.0, 3.0]:
        MODULE.FILTERS[f"mh{int(floor * 100)}_pct{int(pct_cap)}"] = (
            f"least(rank_3d, rank_5d, rank_10d) >= {floor} AND signal_pct_chg_raw <= {pct_cap}"
        )

CASES = [
    {"blend": "w10_100", "score": "plain", "filter": f"mh{int(floor*100)}_pct3", "threshold": 0.98, "topn": 1, "hold": 12}
    for floor in [0.60, 0.65, 0.70, 0.75]
] + [
    {"blend": "w10_100", "score": "plain", "filter": "mh65_pct3", "threshold": threshold, "topn": 1, "hold": 12}
    for threshold in [0.94, 0.95, 0.96]
] + [
    {"blend": "w10_100", "score": "plain", "filter": "mh65_pct3", "threshold": 0.98, "topn": 1, "hold": hold}
    for hold in [8, 10, 15]
] + [
    {"blend": "w10_100", "score": "plain", "filter": "mh65_pct2", "threshold": 0.98, "topn": 1, "hold": 12},
    {"blend": "w10_100", "score": "plain", "filter": "mh65_pct3", "threshold": 0.98, "topn": 2, "hold": 12},
]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODULE.SIGNAL_DIR.mkdir(exist_ok=True)
    MODULE.SCORE_DIR.mkdir(exist_ok=True)
    MODULE.LOG_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(MODULE.FEATURE_DB), read_only=True)
    rows = []
    try:
        score_db = MODULE.build_score(con, "w10_100")
        for case in CASES:
            name, signal_file, max_positions, target_pct = MODULE.build_signal(con, case)
            result = MODULE.run_case(case, name, signal_file, max_positions, target_pct, score_db)
            rows.append(result)
            pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last").to_csv(
                REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig"
            )
            print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (REPORT_DIR / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
