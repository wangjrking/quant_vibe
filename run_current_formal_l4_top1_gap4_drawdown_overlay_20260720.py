from __future__ import annotations

import run_current_formal_l4_top1_drawdown_overlay_20260720 as runner


runner.OUT = runner.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_gap4_drawdown_overlay_20260720"
runner.LOGS = runner.OUT / "logs"
runner.SIGNAL = runner.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top1_open_gap_risk_20260720" / "signals" / "high_gt4_s00.csv"


if __name__ == "__main__":
    runner.main()
