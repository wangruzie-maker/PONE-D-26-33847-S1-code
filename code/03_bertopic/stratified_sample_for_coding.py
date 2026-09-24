# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import sqlite3
import pandas as pd
from sklearn.utils import shuffle
import os

# 定义原始数据库路径、输出数据库路径和临时目录（Windows格式）
db_path = r'os.environ.get("CORPUS_DB", "data/corpus.db")'
output_db_path = r'D:\sampled_data.db'  # 新的数据库文件路径
output_dir = r'D:\切分标注.csv'

# 抽样比例
sampling_ratio = 0.15

# 数据源及其对应的表名
data_sources = {
    'CNN': 'news',
    'NYT': 'my_existing_table',
    'WP': 'wpnews'
}

# 连接到原始SQLite数据库
conn_source = sqlite3.connect(db_path)
cursor = conn_source.cursor()

# 创建或连接到新的SQLite数据库
conn_output = sqlite3.connect(output_db_path)

for source, table_name in data_sources.items():
    # 获取总行数
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]

    # 计算需要抽样的数量
    sample_size = int(total_rows * sampling_ratio)

    # 使用SQL查询进行分层随机抽样
    query = f"""
    SELECT *
    FROM (
        SELECT *, ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
        FROM {table_name}
    )
    WHERE rn <= {sample_size};
    """

    # 执行查询并将结果加载到DataFrame中
    df_sampled = pd.read_sql_query(query, conn_source)

    # 确保有唯一的索引，防止重复
    df_sampled.reset_index(drop=True, inplace=True)

    # 将抽样的数据保存到新的SQLite数据库中的新表里
    df_sampled.to_sql(f'{source}_sample', conn_output, if_exists='replace', index=False)
    print(f"Sample from {source} saved to the new database.")

# 关闭数据库连接
conn_source.close()
conn_output.close()