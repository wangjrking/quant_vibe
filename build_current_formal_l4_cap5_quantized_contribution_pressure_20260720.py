from __future__ import annotations

import build_current_formal_l4_contribution_pressure_20260720 as audit


audit.REPORT = audit.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_quantized_contribution_audit_20260720"
audit.LOG = audit.REPORT / "verbose_full.log"
audit.BASE_SIGNAL = audit.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"
audit.PRESSURE = audit.REPORT / "pressure_signals"


if __name__ == "__main__":
    audit.main()
