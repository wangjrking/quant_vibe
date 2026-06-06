import pandas as pd
import pyarrow.parquet as pq
import os

file_path = 'D:/work/quant/quant001/quant/data_file/stock_factor_data.parquet'

print("Step 1: Check file exists")
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File exists, size: {file_size:.2f} MB")
else:
    print("File not found")
    exit(1)

print("\nStep 2: Read metadata")
try:
    parquet_file = pq.ParquetFile(file_path)
    print(f"Rows: {parquet_file.metadata.num_rows}")
    print(f"Row groups: {parquet_file.metadata.num_row_groups}")
    print(f"Columns: {parquet_file.metadata.num_columns}")
except Exception as e:
    print(f"Failed to read metadata: {e}")
    exit(1)

print("\nStep 3: Try to read first 10 rows")
try:
    df_sample = pd.read_parquet(file_path, nrows=10)
    print(f"Success! Shape: {df_sample.shape}")
    print(f"Columns: {df_sample.columns.tolist()[:10]}")
except Exception as e:
    print(f"Failed: {e}")
    
print("\nStep 4: Try to read with fastparquet")
try:
    df_fast = pd.read_parquet(file_path, engine='fastparquet')
    print(f"Success with fastparquet! Shape: {df_fast.shape}")
except Exception as e:
    print(f"Failed with fastparquet: {e}")

print("\nStep 5: Try to read specific columns")
try:
    parquet_file = pq.ParquetFile(file_path)
    columns = parquet_file.schema_arrow.names
    print(f"Total columns: {len(columns)}")
    
    for i, col in enumerate(columns[:5]):
        try:
            df_col = pd.read_parquet(file_path, columns=[col])
            print(f"Column {i} ({col}): OK")
        except Exception as e:
            print(f"Column {i} ({col}): ERROR - {str(e)[:100]}")
            
except Exception as e:
    print(f"Failed: {e}")

print("\nDiagnosis complete")