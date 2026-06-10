import pandas as pd
import sys

file_path = 'D:/work/quant/quant001/quant/data_file/stock_factor_data.parquet'

print("Attempting to read the full parquet file...")
print("=" * 60)

try:
    df = pd.read_parquet(file_path)
    print("SUCCESS! File read successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")
except Exception as e:
    print(f"FAILED: {e}")
    print(f"Error type: {type(e).__name__}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()