from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build rank-normalized ensembles from prediction fold parquet files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec", action="append", required=True, help="name=prediction_dir=weight")
    parser.add_argument("--base-name", help="Model name whose non-score columns should be retained. Defaults to first spec.")
    return parser.parse_args(argv)


def _parse_spec(text: str) -> tuple[str, Path, float]:
    parts = text.split("=")
    if len(parts) < 3:
        raise ValueError(f"Invalid spec {text!r}; expected name=prediction_dir=weight")
    name = parts[0]
    weight = float(parts[-1])
    prediction_dir = Path("=".join(parts[1:-1]))
    if not prediction_dir.exists():
        raise FileNotFoundError(prediction_dir)
    return name, prediction_dir, weight


def _rank_score(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("trade_date")["pred_prob"].rank(method="average", pct=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    specs = [_parse_spec(text) for text in args.spec]
    total_weight = sum(weight for _, _, weight in specs)
    if total_weight <= 0:
        raise SystemExit("Total weight must be positive.")
    base_name = args.base_name or specs[0][0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_spec = next((spec for spec in specs if spec[0] == base_name), specs[0])
    base_files = sorted(base_spec[1].glob("fold*.parquet"))
    if not base_files:
        raise SystemExit(f"No fold parquet files in {base_spec[1]}")

    manifest = {
        "base_name": base_spec[0],
        "specs": [{"name": name, "prediction_dir": str(path), "weight": weight} for name, path, weight in specs],
        "score_method": "daily cross-sectional rank pct weighted average",
    }
    (output_dir / "ensemble_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for base_file in base_files:
        fold_name = base_file.name
        base = pd.read_parquet(base_file)
        base["trade_date"] = base["trade_date"].astype(str)
        base_keyed = base.set_index(["trade_date", "stock_code"], drop=False)
        ensemble = None
        for name, prediction_dir, weight in specs:
            path = prediction_dir / fold_name
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path, columns=["trade_date", "stock_code", "pred_prob"])
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame[f"rank_{name}"] = _rank_score(frame)
            scores = frame.set_index(["trade_date", "stock_code"])[f"rank_{name}"]
            contribution = scores.reindex(base_keyed.index).fillna(0.0) * (weight / total_weight)
            ensemble = contribution if ensemble is None else ensemble + contribution
            base_keyed[f"pred_{name}"] = scores.reindex(base_keyed.index)
        base_keyed["pred_prob"] = ensemble.astype(float)
        output = base_keyed.reset_index(drop=True)
        output.to_parquet(output_dir / fold_name, index=False)
        print(f"WROTE {output_dir / fold_name} rows={len(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
