from __future__ import annotations

import run_current_formal_l4_cap5_score_state_exit_grid_20260720 as grid


grid.OUT = grid.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_score_state_exit_extreme_20260720"
grid.SIGNALS = grid.OUT / "signals"
grid.LOGS = grid.OUT / "logs"
grid.CASES = [
    {"min_hold": min_hold, "max_hold": 13, "current_max": current_max}
    for min_hold in (1, 2, 4, 6, 8)
    for current_max in (0.01, 0.02, 0.05)
]


if __name__ == "__main__":
    grid.main()
