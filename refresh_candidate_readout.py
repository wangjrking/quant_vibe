from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from evaluate_prediction_asset import (
    _apply_date_filter,
    _auto_label_col,
    _read_parquet_dir,
    _read_sqlite_table,
    evaluate_frame,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Refresh candidate partial evaluation and baseline comparison bundle.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--baseline-json")
    parser.add_argument("--baseline-prediction-db-path")
    parser.add_argument("--baseline-prediction-table")
    parser.add_argument("--baseline-label-col")
    parser.add_argument("--baseline-score-col", default="pred_prob")
    parser.add_argument("--align-baseline-window", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--score-col", default="pred_prob")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--title", default=None)
    return parser.parse_args(argv)


def _split_top_k(raw: str) -> list[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compare(baseline: dict, candidate: dict) -> dict:
    compare_keys = [
        "daily_pearson_ic_mean",
        "daily_rank_ic_mean",
        "rank_ic_positive_ratio",
        "top_decile_mean_return",
        "bottom_decile_mean_return",
        "top_minus_bottom_mean",
        "top_minus_bottom_positive_ratio",
    ]
    deltas = {}
    for key in compare_keys:
        b = baseline.get(key)
        c = candidate.get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            deltas[key] = c - b
    top_deltas = {}
    for key in sorted(set((baseline.get("top_k_mean_returns") or {}).keys()) | set((candidate.get("top_k_mean_returns") or {}).keys()), key=lambda x: int(x)):
        b = (baseline.get("top_k_mean_returns") or {}).get(key)
        c = (candidate.get("top_k_mean_returns") or {}).get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            top_deltas[key] = c - b
    deltas["top_k_mean_returns"] = top_deltas
    return deltas


def _render_markdown(title: str, candidate: dict, deltas: dict) -> str:
    top = deltas.get("top_k_mean_returns", {})
    lines = [
        f"# {title}",
        "",
        "## Candidate Snapshot",
        "",
        f"- source: `{candidate['source']}`",
        f"- label: `{candidate['label_col']}`",
        f"- date range: `{candidate['date_min']}` to `{candidate['date_max']}`",
        f"- trade_days: `{candidate['trade_days']}`",
        f"- valid_rows: `{candidate['valid_rows']}` / `{candidate['total_rows']}`",
        "",
        "## Candidate Metrics",
        "",
        f"- daily_pearson_ic_mean: `{candidate['daily_pearson_ic_mean']:.6f}`",
        f"- daily_rank_ic_mean: `{candidate['daily_rank_ic_mean']:.6f}`",
        f"- rank_ic_positive_ratio: `{candidate['rank_ic_positive_ratio']:.6f}`",
        f"- top_decile_mean_return: `{candidate['top_decile_mean_return']:.6f}`",
        f"- bottom_decile_mean_return: `{candidate['bottom_decile_mean_return']:.6f}`",
        f"- top_minus_bottom_mean: `{candidate['top_minus_bottom_mean']:.6f}`",
        "",
        "## Delta vs Baseline",
        "",
        f"- daily_pearson_ic_mean: `{deltas.get('daily_pearson_ic_mean', 0.0):+.6f}`",
        f"- daily_rank_ic_mean: `{deltas.get('daily_rank_ic_mean', 0.0):+.6f}`",
        f"- rank_ic_positive_ratio: `{deltas.get('rank_ic_positive_ratio', 0.0):+.6f}`",
        f"- top_decile_mean_return: `{deltas.get('top_decile_mean_return', 0.0):+.6f}`",
        f"- top_minus_bottom_mean: `{deltas.get('top_minus_bottom_mean', 0.0):+.6f}`",
        f"- top5_mean_return: `{top.get('5', 0.0):+.6f}`",
        f"- top10_mean_return: `{top.get('10', 0.0):+.6f}`",
        f"- top20_mean_return: `{top.get('20', 0.0):+.6f}`",
    ]
    return "\n".join(lines) + "\n"


def _resolve_baseline_payload(args, candidate_payload: dict) -> tuple[dict, str]:
    if args.baseline_json:
        baseline_json = Path(args.baseline_json)
        return _load_json(baseline_json), str(baseline_json.resolve())
    if args.baseline_prediction_db_path:
        baseline_frame = _read_sqlite_table(Path(args.baseline_prediction_db_path), args.baseline_prediction_table)
        if args.align_baseline_window:
            baseline_frame = _apply_date_filter(
                baseline_frame,
                candidate_payload.get("date_min"),
                candidate_payload.get("date_max"),
            )
        baseline_label_col = args.baseline_label_col or _auto_label_col(baseline_frame, args.baseline_score_col)
        baseline_summary, _ = evaluate_frame(
            baseline_frame,
            score_col=args.baseline_score_col,
            label_col=baseline_label_col,
            top_k=_split_top_k(args.top_k),
            quantiles=args.quantiles,
        )
        payload = {
            "source": f"{Path(args.baseline_prediction_db_path).resolve()}::{args.baseline_prediction_table}",
            "score_col": args.baseline_score_col,
            "label_col": baseline_label_col,
            "date_filter_from": candidate_payload.get("date_min") if args.align_baseline_window else None,
            "date_filter_to": candidate_payload.get("date_max") if args.align_baseline_window else None,
            **baseline_summary,
        }
        return payload, payload["source"]
    raise ValueError("Either --baseline-json or --baseline-prediction-db-path is required")


def main(argv=None) -> int:
    args = parse_args(argv)
    prediction_dir = Path(args.prediction_dir)
    prefix = Path(args.output_prefix)

    frame = _read_parquet_dir(prediction_dir)
    label_col = args.label_col or _auto_label_col(frame, args.score_col)
    summary, _daily = evaluate_frame(
        frame,
        score_col=args.score_col,
        label_col=label_col,
        top_k=_split_top_k(args.top_k),
        quantiles=args.quantiles,
    )
    candidate_payload = {
        "source": str(prediction_dir.resolve()),
        "score_col": args.score_col,
        "label_col": label_col,
        **summary,
    }
    baseline, baseline_source = _resolve_baseline_payload(args, candidate_payload)
    deltas = _compare(baseline, candidate_payload)
    compare_payload = {
        "baseline": baseline_source,
        "candidate": str((prefix.parent / (prefix.name + ".eval.json")).resolve()),
        "deltas": deltas,
    }

    eval_path = prefix.parent / f"{prefix.name}.eval.json"
    baseline_path = prefix.parent / f"{prefix.name}.baseline.json"
    compare_path = prefix.parent / f"{prefix.name}.compare.json"
    md_path = prefix.parent / f"{prefix.name}.md"
    _write_json(eval_path, candidate_payload)
    _write_json(baseline_path, baseline)
    _write_json(compare_path, compare_payload)
    md_path.write_text(
        _render_markdown(args.title or prefix.name, candidate_payload, deltas),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
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
