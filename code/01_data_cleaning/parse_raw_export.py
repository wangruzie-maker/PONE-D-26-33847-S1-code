# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import re
import csv

def read_file(file_path, encodings=['utf-8', 'gbk', 'gb18030']):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the file with the provided encodings")

def clean_text(text):
    # 剔除所有中文内容及其标点符号
    text = re.sub(r'[\u4e00-\u9fff\u3000-\u303f]+', '', text)
    # 剔除过长的虚线
    text = re.sub(r'-{10,}', '', text)
    # 剔除日期后的多余内容
    text = re.sub(r'\b(Most Recent|Economy)\b.*', '', text, flags=re.IGNORECASE)
    # 剔除正文内的链接
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 剔除正文部分的文章编号后的数字串
    text = re.sub(r'\b\d+\.\s+', '', text)
    # 剔除正文部分开头的分类标签
    text = re.sub(r'~r/rss/cnn_latest/~3/[a-zA-Z0-9_\-]+/index\.html\s*', '', text)
    text = re.sub(r'(London|NEW YORK)\s*\([a-zA-Z\s]+\)\s*', '', text)
    text = re.sub(r'(Business|Investing|Top Fortune Stories|STORY HIGHLIGHTS)\s*', '', text)
    # 去除多余的空格，保留单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_news(file_path):
    news_list = []
    content, encoding = read_file(file_path)

    # 使用正则表达式匹配每条新闻
    pattern = re.compile(
        r'(\d+\.\s+)(.*?)(\s+-\s+\()(.*?)(\))\s+(\d{4}-\d{2}-\d{2})\s*(.*?)(?=^\d+\.\s+|\Z)',
        re.DOTALL | re.MULTILINE
    )
    matches = pattern.findall(content)

    for match in matches:
        title = match[3].strip()  # 标题
        date = match[5].strip()  # 日期
        body = match[6].strip()  # 完整正文

        # 清洗正文
        body = clean_text(body)

        # 合并分割的正文内容
        while body.endswith('...'):
            next_match = pattern.search(content, pos=pattern.endpos)
            if not next_match:
                break
            body += next_match.group(7).strip()

        # 验证新闻条目的完整性
        if all([title, date, body]):
            news_list.append({
                'title': title,
                'date': date,
                'body': body
            })

    return news_list

# 调用函数并保存到CSV文件
news_items = extract_news('E:/wpall.txt')

# 确保生成的CSV文件包含首行在内仅有551行
if len(news_items) + 1 != 1260:
    raise ValueError(f"Expected 550 news items plus header, but found {len(news_items)}")

# 定义CSV文件路径
csv_file_path = 'E:/wpdata.csv'

# 写入CSV文件
with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
    fieldnames = ['title', 'date', 'body']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in news_items:
        writer.writerow(item)

print(f"数据已成功保存到 {csv_file_path}")