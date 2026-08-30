from __future__ import annotations

import run_current_formal_l4_top1_open_gap_risk_grid_20260720 as runner


runner.OUT = runner.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_open_gap_reject_neighborhood_20260720"
runner.SIGNALS = runner.OUT / "signals"
runner.LOGS = runner.OUT / "logs"
runner.CASES = [
    {
        "name": f"high_gt{str(threshold).replace('.', 'p')}_reject",
        "mode": "high",
        "threshold": threshold,
        "scale": 0.0,
    }
    for threshold in (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.0)
]


if __name__ == "__main__":
    runner.main()
