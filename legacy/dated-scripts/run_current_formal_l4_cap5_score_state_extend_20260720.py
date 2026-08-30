from __future__ import annotations

import run_current_formal_l4_cap5_score_state_exit_grid_20260720 as grid


grid.OUT = grid.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_score_state_extend_20260720"
grid.SIGNALS = grid.OUT / "signals"
grid.LOGS = grid.OUT / "logs"
grid.CASES = [
    {"min_hold": 12, "max_hold": max_hold, "current_max": current_max}
    for max_hold in (14, 15, 16)
    for current_max in (0.30, 0.50, 0.70, 0.85, 0.95)
]


if __name__ == "__main__":
    grid.main()
