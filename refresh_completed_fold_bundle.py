from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from refresh_candidate_readout import (
    _compare,
    _render_markdown,
    _resolve_baseline_payload,
    _split_top_k,
    _write_json,
)
from evaluate_prediction_asset import _auto_label_col, _read_parquet_dir, evaluate_frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Refresh a partial candidate bundle using all completed folds in a run directory.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--baseline-json")
    parser.add_argument("--baseline-prediction-db-path")
    parser.add_argument("--baseline-prediction-table")
    parser.add_argument("--baseline-label-col")
    parser.add_argument("--baseline-score-col", default="pred_prob")
    parser.add_argument("--align-baseline-window", action="store_true")
    parser.add_argument("--score-col", default="pred_prob")
    parser.add_argument("--label-col")
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--title", default=None)
    return parser.parse_args(argv)


def _read_completed_folds(run_dir: Path) -> list[int]:
    fold_results = run_dir / "fold_results.csv"
    if not fold_results.exists():
        return []
    completed: list[int] = []
    with fold_results.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") in {"ok", "skipped_existing"}:
                try:
                    completed.append(int(row["fold"]))
                except Exception:
                    continue
    return sorted(set(completed))


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.prediction_dir)
    completed_folds = _read_completed_folds(run_dir)
    if not completed_folds:
        raise ValueError(f"No completed folds found under {run_dir}")

    frame = _read_parquet_dir(run_dir)
    label_col = args.label_col or _auto_label_col(frame, args.score_col)
    summary, _ = evaluate_frame(
        frame,
        score_col=args.score_col,
        label_col=label_col,
        top_k=_split_top_k(args.top_k),
        quantiles=args.quantiles,
    )
    candidate_payload = {
        "source": str(run_dir.resolve()),
        "score_col": args.score_col,
        "label_col": label_col,
        "completed_folds": completed_folds,
        "completed_fold_count": len(completed_folds),
        **summary,
    }
    baseline, baseline_source = _resolve_baseline_payload(args, candidate_payload)
    deltas = _compare(baseline, candidate_payload)

    prefix = Path(args.output_prefix)
    eval_path = prefix.parent / f"{prefix.name}.eval.json"
    baseline_path = prefix.parent / f"{prefix.name}.baseline.json"
    compare_path = prefix.parent / f"{prefix.name}.compare.json"
    md_path = prefix.parent / f"{prefix.name}.md"

    compare_payload = {
        "baseline": baseline_source,
        "candidate": str(eval_path.resolve()),
        "completed_folds": completed_folds,
        "deltas": deltas,
    }

    _write_json(eval_path, candidate_payload)
    _write_json(baseline_path, baseline)
    _write_json(compare_path, compare_payload)
    title = args.title or f"{prefix.name} ({len(completed_folds)} folds)"
    md_path.write_text(_render_markdown(title, candidate_payload, deltas), encoding="utf-8")

    print(
        json.dumps(
            {
                "completed_folds": completed_folds,
                "eval_json": str(eval_path.resolve()),
                "baseline_json": str(baseline_path.resolve()),
                "compare_json": str(compare_path.resolve()),
                "markdown": str(md_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
