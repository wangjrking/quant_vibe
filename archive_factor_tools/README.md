# L3 Factor Tool Archive

Archived from `quant/main/` on `2026-06-19` during L3 factor-layer file
governance cleanup.

These files were removed from the main script directory because they are not
part of the current standard production factor chain entrypoints. They are
retained here only as historical repair / diagnosis / reference artifacts.

Archived files:

- `_tmp_GTJA_Alpha191_reference.py`
- `diagnose_all_columns.py`
- `diagnose_parquet.py`
- `diagnose_parquet_simple.py`
- `repair_production_factor_aux_columns.py`

Current policy:

- Standard-chain factor updates should continue to use maintained entrypoints
  such as `incremental_factor_update_target_date.py`,
  `build_production_factor_parts.py`, and the approved rebuild scripts.
- Historical compatibility helpers that still have tests or explicit governance
  value remain in `quant/main/` until a broader legacy-tool reorganization is
  approved.
