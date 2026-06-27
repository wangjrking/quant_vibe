from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize top-rank tuning batch progress.")
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def _read_fold_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _tail_text(path: Path, lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return text[-lines:]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(candidate["output_dir"])
    fold_results = _read_fold_results(output_dir / "fold_results.csv")
    ok_rows = [row for row in fold_results if row.get("status") in {"ok", "skipped_existing"}]
    failed_rows = [row for row in fold_results if row.get("status") == "failed"]
    latest_file = None
    latest_ts = None
    if output_dir.exists():
        for file in output_dir.rglob("*"):
            if file.is_file():
                ts = file.stat().st_mtime
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                    latest_file = file
    fold_log = output_dir / "fold_logs" / "fold01.log"
    latest_model_metadata = None
    if output_dir.exists():
        metadata_files = sorted((output_dir / "models").glob("model_fold*_metadata.json")) if (output_dir / "models").exists() else []
        if metadata_files:
            latest_model_metadata = _read_json(metadata_files[-1])
    return {
        "label": candidate["label"],
        "candidate_id": candidate["candidate_id"],
        "output_dir": str(output_dir),
        "selected_feature_count_nominal": candidate.get("selected_feature_count_nominal"),
        "selected_feature_count_effective_current_chain": candidate.get("selected_feature_count_effective_current_chain"),
        "actual_model_feature_count_latest": len((latest_model_metadata or {}).get("feature_columns", [])),
        "actual_model_eval_metric_latest": ((latest_model_metadata or {}).get("xgb_params", {}) or {}).get("eval_metric"),
        "planned_folds": 9,
        "completed_folds": len(ok_rows),
        "failed_folds": [int(row["fold"]) for row in failed_rows if row.get("fold")],
        "latest_file": str(latest_file) if latest_file else None,
        "fold01_log_tail": _tail_text(fold_log, 12),
        "fold_results_exists": (output_dir / "fold_results.csv").exists(),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    payload = {
        "report_root": str(Path(args.report_root).resolve()),
        "manifest_path": str(Path(args.manifest).resolve()),
        "candidates": [summarize_candidate(candidate) for candidate in manifest.get("candidates", [])],
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
