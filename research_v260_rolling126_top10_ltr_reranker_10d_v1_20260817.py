"""Build the single fixed 126-date rolling Top10 reranker candidate."""
from __future__ import annotations

from pathlib import Path

import research_v260_rolling252_top10_ltr_reranker_10d_v1_20260817 as core


core.CONTRACT = Path(
    "quant/data_file/runtime/agent_workspaces/strategy-agent/work/"
    "v260_rolling126_top10_ltr_reranker_20260817/training_contract.json"
)
core.OUT = Path(
    "quant/data_file/reports/model_agent_v260_rolling126_top10_ltr_reranker_10d_v1_20260817_r1"
)
core.WINDOW = 126


if __name__ == "__main__":
    raise SystemExit(core.main())
