from __future__ import annotations

import validate_amt9p5w_scale7454_current_20260625 as base


base.REPORT_DIR = base.DATA / "reports" / "strategy_agent_amt9p5w_pos895_scale7454_current_validation_20260625"
base.SIGNAL_FILE = base.DATA / "reports" / "strategy_agent_amt9p5w_true_target_20260625" / "pos895.csv"
base.TARGET_PCT = 0.895
base.DD_SOFT_SCALE = 0.74
base.DD_HARD_SCALE = 0.54
base.BASE_ENV["GM_EQUITY_DD_SOFT_SCALE"] = str(base.DD_SOFT_SCALE)
base.BASE_ENV["GM_EQUITY_DD_HARD_SCALE"] = str(base.DD_HARD_SCALE)


if __name__ == "__main__":
    raise SystemExit(base.main())
