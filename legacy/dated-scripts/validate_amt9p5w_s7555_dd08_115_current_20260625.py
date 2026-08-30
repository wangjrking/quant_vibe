from __future__ import annotations

import validate_amt9p5w_scale7454_current_20260625 as base


base.REPORT_DIR = base.DATA / "reports" / "strategy_agent_amt9p5w_s7555_dd08_115_current_validation_20260625"
base.SIGNAL_FILE = (
    base.DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv"
)
base.TARGET_PCT = 0.8975
base.DD_SOFT = 0.08
base.DD_HARD = 0.115
base.DD_RECOVER = 0.03
base.DD_SOFT_SCALE = 0.75
base.DD_HARD_SCALE = 0.55
base.BASE_ENV["GM_EQUITY_DD_SOFT_TRIGGER"] = str(base.DD_SOFT)
base.BASE_ENV["GM_EQUITY_DD_HARD_TRIGGER"] = str(base.DD_HARD)
base.BASE_ENV["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(base.DD_RECOVER)
base.BASE_ENV["GM_EQUITY_DD_SOFT_SCALE"] = str(base.DD_SOFT_SCALE)
base.BASE_ENV["GM_EQUITY_DD_HARD_SCALE"] = str(base.DD_HARD_SCALE)


if __name__ == "__main__":
    raise SystemExit(base.main())
