# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import sqlite3
import re

# 绝对路径
db_path = os.environ.get("CORPUS_DB", "data/corpus.db")

# 连接到数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


# 定义数据清洗函数
def clean_text(text):
    # 剔除正文末尾特定的句子
    end_sentences = [
        r'Thank you for your patience while we verify access\. If you are in Reader mode please exit and log into your Times account, or subscribe for all of The Times\. Thank you for your patience while we verify access\. Already a subscriber\? Log in\. Want all of The Times\? Subscribe\.',
        r': $\d+$ $""$ $""$'
    ]

    for sentence in end_sentences:
        text = re.sub(sentence, '', text)

    # 剔除每一行标题最后的 "- The New York Times" 字符串
    text = re.sub(r' - The New York Times', '', text)

    # 去除正文单词出现之前的多余“：”
    text = re.sub(r'^:\s*', '', text, flags=re.MULTILINE)

    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 创建备份表
cursor.execute("CREATE TABLE IF NOT EXISTS articles_backup AS SELECT * FROM news")

# 插入备份数据
cursor.execute("INSERT INTO articles_backup SELECT * FROM my_existing_table")

# 提交更改
conn.commit()

# 读取并清洗数据
cursor.execute("SELECT body FROM my_existing_table")
rows = cursor.fetchall()

cleaned_rows = [clean_text(row[0]) for row in rows]

# 清空表
cursor.execute("DELETE FROM my_existing_table")

# 插入清洗后的数据
insert_query = "INSERT INTO my_existing_table (body) VALUES (?)"
cursor.executemany(insert_query, [(row,) for row in cleaned_rows])

# 提交更改
conn.commit()

# 关闭连接
cursor.close()
conn.close()