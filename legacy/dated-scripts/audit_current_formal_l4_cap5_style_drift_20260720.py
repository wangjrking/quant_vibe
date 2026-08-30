from __future__ import annotations

import audit_current_formal_l4_style_drift_20260720 as audit


audit.REPORT = audit.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_contribution_audit_20260720"
audit.TRADES = audit.REPORT / "trade_contributions.csv"
audit.SIGNAL = audit.ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cap5_rank_hold_refine_20260720" / "signals" / "r1_88_min12_max13.csv"


if __name__ == "__main__":
    audit.main()
