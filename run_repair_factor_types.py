from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import resolve_data_dir


DATA_DIR = resolve_data_dir()
FACTOR_PATH = DATA_DIR / "stock_factor_data.parquet"
TEXT_COLUMNS = {"stock_code", "trade_date", "name", "industry", "act_ent_type"}


def main() -> None:
    print("repair_factor_types_start", flush=True)
    frame = pd.read_parquet(FACTOR_PATH)
    frame["trade_date"] = frame["trade_date"].astype(str)

    if "st_type" in frame.columns:
        st_as_text = frame["st_type"].astype("string")
        frame["st_type"] = np.where(st_as_text.eq("ST").fillna(False), 1.0, 0.0)

    for col in frame.columns:
        if col in TEXT_COLUMNS or col == "st_type":
            continue
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.to_parquet(FACTOR_PATH)
    print(
        f"repair_factor_types_done rows={frame.shape[0]} "
        f"max={frame['trade_date'].max()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
