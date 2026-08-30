from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(r"D:\work\quant\quant_mcp\quant\main\tune_current_prod_four_year_formal_20260628.py")

spec = importlib.util.spec_from_file_location("tune_mod", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    summary_path = mod.REPORT_DIR / "summary.csv"
    detail_path = mod.REPORT_DIR / "detail.csv"
    summary_rows = mod._read_rows(summary_path)
    detail_rows = mod._read_rows(detail_path)
    done = {row["case_key"] for row in summary_rows if row.get("case_key")}

    for entry in mod.ENTRY_CASES:
        for exe in mod.EXEC_CASES:
            case_key = mod._case_key(entry, exe)
            signal_file = mod.REPORT_DIR / "signals" / f"{case_key}.csv"
            if case_key in done or not signal_file.exists():
                continue
            case = {
                "case_key": case_key,
                "signal_file": signal_file,
                "score_table": mod._score_table(case_key),
                "signal_rows": len(mod._read_rows(signal_file)),
            }
            rows = [mod._run_backtest(case, exe, tag, start, end) for tag, start, end in mod.TIME_SLICES]
            detail_rows.extend(rows)
            summary_rows.append(mod._summary_row(entry, exe, case, rows))
            summary_rows.sort(key=mod._sort_key, reverse=True)
            mod._write_rows(detail_path, detail_rows)
            mod._write_rows(summary_path, summary_rows)
            mod._write_json(mod.REPORT_DIR / "summary.json", summary_rows)
            done.add(case_key)
            print(f"completed {case_key}", flush=True)


if __name__ == "__main__":
    main()
