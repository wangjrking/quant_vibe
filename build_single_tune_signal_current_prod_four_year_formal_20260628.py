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
    args = parser.parse_args()

    manifest, _, sources = mod._load_strategy(mod.STRATEGY_DIR)
    entry = next(item for item in mod.ENTRY_CASES if item["name"] == args.entry)
    exe = next(item for item in mod.EXEC_CASES if item["name"] == args.exe)
    case = mod._build_case_assets(manifest, sources, entry, exe)
    print(case["case_key"])


if __name__ == "__main__":
    main()
