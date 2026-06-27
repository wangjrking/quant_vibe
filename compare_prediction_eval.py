from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare a candidate evaluation JSON against a baseline evaluation JSON.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    args = parse_args(argv)
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    compare_keys = [
        "daily_pearson_ic_mean",
        "daily_rank_ic_mean",
        "rank_ic_positive_ratio",
        "top_decile_mean_return",
        "bottom_decile_mean_return",
        "top_minus_bottom_mean",
        "top_minus_bottom_positive_ratio",
    ]
    result = {
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "deltas": {},
    }
    for key in compare_keys:
        b = baseline.get(key)
        c = candidate.get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            result["deltas"][key] = c - b
    baseline_top = baseline.get("top_k_mean_returns", {})
    candidate_top = candidate.get("top_k_mean_returns", {})
    top_deltas = {}
    for key in sorted(set(baseline_top) | set(candidate_top), key=lambda x: int(x)):
        b = baseline_top.get(key)
        c = candidate_top.get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            top_deltas[key] = c - b
    result["deltas"]["top_k_mean_returns"] = top_deltas
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
