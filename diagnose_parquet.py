import pandas as pd
import pyarrow.parquet as pq
import os
from project_paths import resolve_data_dir

file_path = resolve_data_dir() / 'stock_factor_data.parquet'

print("=" * 60)
print("Parquet 文件诊断")
print("=" * 60)

# 1. 检查文件基本信息
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"[OK] 文件存在")
    print(f"[OK] 文件大小: {file_size:.2f} MB")
else:
    print("[ERROR] 文件不存在")
    exit(1)

# 2. 尝试读取文件元数据
print("\n" + "=" * 60)
print("尝试读取文件元数据...")
print("=" * 60)
try:
    parquet_file = pq.ParquetFile(file_path)
    print(f"[OK] 文件元数据读取成功")
    print(f"[OK] 文件行数: {parquet_file.metadata.num_rows}")
    print(f"[OK] 行组数量: {parquet_file.metadata.num_row_groups}")
    print(f"[OK] 列数: {parquet_file.metadata.num_columns}")
    print(f"\n前10列信息:")
    for i in range(min(10, parquet_file.metadata.num_columns)):
        col_name = parquet_file.schema_arrow.names[i]
        col_type = parquet_file.schema_arrow.types[i]
        print(f"  - {col_name}: {col_type}")
except Exception as e:
    print(f"[ERROR] 读取元数据失败: {e}")

# 3. 尝试读取部分数据
print("\n" + "=" * 60)
print("尝试读取前10行数据...")
print("=" * 60)
try:
    df_sample = pd.read_parquet(file_path, nrows=10)
    print(f"[OK] 成功读取前10行")
    print(f"[OK] 数据形状: {df_sample.shape}")
    print(f"\n列名:")
    print(df_sample.columns.tolist()[:10])  # 只显示前10列
except Exception as e:
    print(f"[ERROR] 读取数据失败: {e}")

# 4. 尝试逐列读取找出问题列
print("\n" + "=" * 60)
print("逐列读取测试（显示前20列）...")
print("=" * 60)
try:
    parquet_file = pq.ParquetFile(file_path)
    columns = parquet_file.schema_arrow.names

    problematic_columns = []
    for i, col in enumerate(columns[:20]):  # 只测试前20列
        try:
            df_col = pd.read_parquet(file_path, columns=[col])
            print(f"[OK] {col}: OK")
        except Exception as e:
            print(f"[ERROR] {col}: {str(e)[:100]}")
            problematic_columns.append(col)

    if problematic_columns:
        print(f"\n问题列: {problematic_columns}")
    else:
        print("\n[OK] 前20列都可以正常读取")

except Exception as e:
    print(f"[ERROR] 逐列读取测试失败: {e}")

# 5. 尝试使用不同的引擎读取
print("\n" + "=" * 60)
print("尝试使用不同引擎读取...")
print("=" * 60)
try:
    print("使用 pyarrow 引擎...")
    df_pyarrow = pd.read_parquet(file_path, engine='pyarrow')
    print(f"[OK] pyarrow 引擎读取成功，数据形状: {df_pyarrow.shape}")
except Exception as e:
    print(f"[ERROR] pyarrow 引擎失败: {str(e)[:200]}")

try:
    print("\n使用 fastparquet 引擎...")
    df_fast = pd.read_parquet(file_path, engine='fastparquet')
    print(f"[OK] fastparquet 引擎读取成功，数据形状: {df_fast.shape}")
except Exception as e:
    print(f"[ERROR] fastparquet 引擎失败: {str(e)[:200]}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
print("\n建议解决方案:")
print("1. 如果文件损坏，重新运行 data_process_module.py 生成新文件")
print("2. 如果特定列有问题，修改读取代码排除这些列")
print("3. 尝试使用 fastparquet 引擎读取")
