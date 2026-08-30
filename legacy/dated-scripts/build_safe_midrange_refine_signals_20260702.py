from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
MID_DIR = BASE_DIR / "postrank_open_filter_candidates" / "safe_midrange_candidates"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "safe_midrange_refine_candidates"


CASES = [
    {"name": "lowflat12_top6_p16", "source": "lowflat12_top6_p15", "target_pct": "0.16000"},
    {"name": "lowflat12_top6_p17", "source": "lowflat12_top6_p15", "target_pct": "0.17000"},
    {"name": "lowflat12_top6_p18", "source": "lowflat12_top6_p15", "target_pct": "0.18000"},
    {"name": "lowflat10_top6_p16", "source": "lowflat10_top6_p15", "target_pct": "0.16000"},
    {"name": "lowflat10_top6_p17", "source": "lowflat10_top6_p15", "target_pct": "0.17000"},
    {"name": "nohigh12_top7_p13", "source": "nohigh12_top7_p12", "target_pct": "0.13000"},
    {"name": "nohigh12_top7_p14", "source": "nohigh12_top7_p12", "target_pct": "0.14000"},
    {"name": "nohigh12_top8_p11", "source": "nohigh12_top7_p12", "target_pct": "0.11000"},
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        source = MID_DIR / case["source"] / "signals.csv"
        df = pd.read_csv(source, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
        if case["name"] == "nohigh12_top8_p11":
            # Use the original Top8 source for this one; the Top7 source is too narrow.
            source = BASE_DIR / "postrank_open_filter_candidates" / "sharpe_focus_candidates" / "nohigh_pct12_top8" / "signals.csv"
            df = pd.read_csv(source, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
        df["target_pct"] = case["target_pct"]
        df["holding_days"] = 1
        df["max_holding_days"] = 1
        df["score_continue_entry_ratio"] = "9.99000"
        df["strategy_variant"] = case["name"]
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        df.to_csv(signal_file, index=False, encoding="utf-8")
        counts = df.groupby("signal_date").size()
        rows.append(
            {
                "name": case["name"],
                "source": str(source),
                "signal_file": str(signal_file),
                "rows": int(len(df)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()),
                "target_pct": case["target_pct"],
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
