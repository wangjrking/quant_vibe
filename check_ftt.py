import pandas as pd

df = pd.read_parquet('D:/办公/量化交易/quant_project/data_file/ftt_test_data.parquet')
print("Shape:", df.shape)
print("\nColumns (first 30):")
print(df.columns.tolist()[:30])
print("\nFTT columns:")
ftt_cols = [col for col in df.columns if col.startswith('ftt_')]
print(ftt_cols)
print("\nFirst 3 rows:")
print(df.head(3))
