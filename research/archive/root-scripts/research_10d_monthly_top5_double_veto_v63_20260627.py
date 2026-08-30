from __future__ import annotations

from pathlib import Path

import research_10d_monthly_top5_double_veto_v62_20260627 as v62


ROOT = Path(__file__).resolve().parents[2]

v62.REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_10d_monthly_top5_double_veto_v63_20260627"
v62.TARGET_TABLE = "stock_predict_data_model_agent_10d_monthly_top5_double_veto_v63_20260627_executable_10d_open_return_research"


if __name__ == "__main__":
    raise SystemExit(v62.main())
