import pandas as pd
import pyarrow.parquet as pq
import sys

file_path = 'D:/work/quant/quant001/quant/data_file/stock_factor_data.parquet'

print("Testing all columns to find problematic ones...")
print("=" * 60)

try:
    parquet_file = pq.ParquetFile(file_path)
    columns = parquet_file.schema_arrow.names
    total_cols = len(columns)
    
    print(f"Total columns to test: {total_cols}")
    
    problematic_columns = []
    successful_columns = []
    
    for i, col in enumerate(columns):
        try:
            df_col = pd.read_parquet(file_path, columns=[col])
            successful_columns.append(col)
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{total_cols} columns tested, {len(problematic_columns)} errors found")
        except Exception as e:
            error_msg = str(e)[:150]
            print(f"ERROR in column {i} ({col}): {error_msg}")
            problematic_columns.append((col, error_msg))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total columns: {total_cols}")
    print(f"Successful: {len(successful_columns)}")
    print(f"Failed: {len(problematic_columns)}")
    
    if problematic_columns:
        print("\nProblematic columns:")
        for col, error in problematic_columns:
            print(f"  - {col}: {error}")
    else:
        print("\nAll columns can be read successfully!")
        
    print("\n" + "=" * 60)
    print("Testing full file read...")
    print("=" * 60)
    
    try:
        df = pd.read_parquet(file_path)
        print(f"SUCCESS! Full file read completed.")
        print(f"Shape: {df.shape}")
    except Exception as e:
        print(f"FAILED to read full file: {str(e)[:200]}")
        print("\nTrying to read without problematic columns...")
        
        if problematic_columns:
            good_cols = [c for c in columns if c not in [p[0] for p in problematic_columns]]
            try:
                df = pd.read_parquet(file_path, columns=good_cols)
                print(f"SUCCESS! Read file without problematic columns.")
                print(f"Shape: {df.shape}")
                print(f"Excluded {len(problematic_columns)} problematic columns")
            except Exception as e2:
                print(f"FAILED even without problematic columns: {str(e2)[:200]}")
        
except Exception as e:
    print(f"Fatal error: {e}")
    sys.exit(1)