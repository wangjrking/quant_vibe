from __future__ import annotations

import validate_entry_confirm_best_20260624 as base
import validate_force_sell_pos56_candidate_20260624 as validator


CASE_NAME = "entry_confirm_w78_5d12_3d10_pos56_h3_e097_mh1"
REPORT_DIR = (
    validator.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "entry_confirm_w78_5d12_3d10_pos56_h3_e097_validation_20260624"
)
SIGNAL_FILE = (
    validator.SOURCE_DIR
    / "formal_horizon_entry_confirmation_20260624"
    / "entry_confirm_exit_neighborhood_20260624"
    / "signals"
    / f"{CASE_NAME}.csv"
)


def main() -> int:
    old_case = validator.CASE_NAME
    old_report = validator.REPORT_DIR
    old_signal = validator.SIGNAL_FILE
    old_exit = validator.EXIT_RATIO
    old_base_case = base.CASE_NAME
    old_base_report = base.REPORT_DIR
    old_base_signal = base.SIGNAL_FILE
    try:
        validator.CASE_NAME = CASE_NAME
        validator.REPORT_DIR = REPORT_DIR
        validator.SIGNAL_FILE = SIGNAL_FILE
        validator.EXIT_RATIO = 0.97
        base.CASE_NAME = CASE_NAME
        base.REPORT_DIR = REPORT_DIR
        base.SIGNAL_FILE = SIGNAL_FILE
        result = validator.main()
        base._write_clean_report()
        return result
    finally:
        validator.CASE_NAME = old_case
        validator.REPORT_DIR = old_report
        validator.SIGNAL_FILE = old_signal
        validator.EXIT_RATIO = old_exit
        base.CASE_NAME = old_base_case
        base.REPORT_DIR = old_base_report
        base.SIGNAL_FILE = old_base_signal


if __name__ == "__main__":
    raise SystemExit(main())
