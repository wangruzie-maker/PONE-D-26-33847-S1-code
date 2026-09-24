# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os

import pandas as pd
import sqlite3

# 尝试不同的编码方式
encodings = ['utf-8', 'latin1', 'cp1252', 'gbk']

# 读取CSV文件
for encoding in encodings:
    try:
        df = pd.read_csv('E:/wpdata.csv', encoding=encoding)
        print(f"Successfully read the file with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the file with encoding: {encoding}")
else:
    print("None of the provided encodings worked. Please check the file encoding.")
    exit(1)

# 连接到SQLite数据库
conn = sqlite3.connect('os.environ.get("CORPUS_DB", "data/corpus.db")')
cursor = conn.cursor()

# 将数据插入到现有的SQLite表格中
df.to_sql('wpnews', conn, if_exists='append', index=False)

# 验证数据是否成功插入
cursor.execute("SELECT * FROM my_existing_table LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)

# 提交事务并关闭连接
conn.commit()
conn.close()

