from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


MODULE_PATH = Path(r"D:\work\quant\quant_mcp\quant\main\validate_w80_ddtight_lowpath_20260628.py")

spec = importlib.util.spec_from_file_location("lowpath_mod", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--report-name", required=True)
    args = parser.parse_args()

    mod.CASE_NAME = args.case_name
    mod.SIGNAL_FILE = mod.BASE_REPORT / "signals" / f"{args.case_name}.csv"
    mod.SCORE_TABLE = f"score_{args.case_name}"
    mod.REPORT_DIR = mod.DATA / "reports" / args.report_name
    mod.main()


if __name__ == "__main__":
    main()
