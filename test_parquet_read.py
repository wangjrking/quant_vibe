import pandas as pd
import sys
from project_paths import resolve_data_dir

file_path = resolve_data_dir() / 'stock_factor_data.parquet'

print("Test 1: Direct read")
try:
    df = pd.read_parquet(file_path)
    print(f"SUCCESS! Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    sys.exit(0)
except Exception as e:
    print(f"FAILED: {e}")

    print("\nTest 2: Read with error handling")
    import pyarrow.parquet as pq

    try:
        parquet_file = pq.ParquetFile(file_path)
        columns = parquet_file.schema_arrow.names

        good_cols = []
        bad_cols = []

        print(f"Testing {len(columns)} columns...")

        for i, col in enumerate(columns):
            try:
                pd.read_parquet(file_path, columns=[col])
                good_cols.append(col)
            except:
                bad_cols.append(col)

            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(columns)}, Errors: {len(bad_cols)}")

        print(f"\nGood columns: {len(good_cols)}")
        print(f"Bad columns: {len(bad_cols)}")

        if bad_cols:
            print(f"\nBad columns list: {bad_cols}")

        print("\nReading file with good columns...")
        df = pd.read_parquet(file_path, columns=good_cols)
        print(f"SUCCESS! Shape: {df.shape}")

    except Exception as e2:
        print(f"FAILED: {e2}")
        sys.exit(1)
