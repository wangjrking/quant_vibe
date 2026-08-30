from __future__ import annotations

import run_current_formal_l4_contribution_pressure_juejin_20260720 as runner


runner.OUT = runner.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_quantized_contribution_audit_20260720"
runner.SIGNALS = runner.OUT / "pressure_signals"
runner.LOGS = runner.OUT / "pressure_logs"
runner.STRATEGY_DIR = runner.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_score_quantized_20260720" / "research_code_snapshot"


if __name__ == "__main__":
    runner.main()
