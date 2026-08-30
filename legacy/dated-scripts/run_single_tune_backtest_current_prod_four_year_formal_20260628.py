from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


MODULE_PATH = Path(r"D:\work\quant\quant_mcp\quant\main\tune_current_prod_four_year_formal_20260628.py")

spec = importlib.util.spec_from_file_location("tune_mod", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--slice", required=True)
    args = parser.parse_args()

    entry = next(item for item in mod.ENTRY_CASES if item["name"] == args.entry)
    exe = next(item for item in mod.EXEC_CASES if item["name"] == args.exe)
    tag, start, end = next(item for item in mod.TIME_SLICES if item[0] == args.slice)
    case_key = mod._case_key(entry, exe)
    case = {
        "case_key": case_key,
        "signal_file": mod.REPORT_DIR / "signals" / f"{case_key}.csv",
        "score_table": mod._score_table(case_key),
        "signal_rows": len(mod._read_rows(mod.REPORT_DIR / "signals" / f"{case_key}.csv")),
    }
    print(mod._run_backtest(case, exe, tag, start, end))


if __name__ == "__main__":
    main()
