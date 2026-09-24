# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import sqlite3


def remove_duplicates(source_db_path, target_db_path, table_name):
    # 连接到源数据库
    source_conn = sqlite3.connect(source_db_path)
    source_cursor = source_conn.cursor()

    # 连接到目标数据库（如果不存在则创建）
    target_conn = sqlite3.connect(target_db_path)
    target_cursor = target_conn.cursor()

    # 创建目标表（如果不存在）
    source_cursor.execute(f'PRAGMA table_info({table_name})')
    columns = source_cursor.fetchall()
    column_definitions = ', '.join([f'{col[1]} {col[2]}' for col in columns])
    target_cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})')

    # 获取所有唯一的标题
    source_cursor.execute(f'SELECT DISTINCT title FROM {table_name}')
    unique_titles = [row[0] for row in source_cursor.fetchall()]

    # 对于每个唯一的标题，保留最早的一条记录，插入到目标表中
    for title in unique_titles:
        source_cursor.execute(f'''
            SELECT * FROM {table_name}
            WHERE title = ? AND rowid = (
                SELECT MIN(rowid)
                FROM {table_name}
                WHERE title = ?
            )
        ''', (title, title))

        # 插入到目标表中
        row = source_cursor.fetchone()
        if row:
            placeholders = ', '.join(['?'] * len(row))
            target_cursor.execute(f'INSERT INTO {table_name} VALUES ({placeholders})', row)

    # 提交事务
    target_conn.commit()

    # 关闭连接
    source_conn.close()
    target_conn.close()


# 使用函数，指定源数据库路径、目标数据库路径和表名
source_db_path = 'os.environ.get("CORPUS_DB", "data/corpus.db")'
target_db_path = 'E:/datanews.db'
table_name = 'wpnews'
remove_duplicates(source_db_path, target_db_path, table_name)