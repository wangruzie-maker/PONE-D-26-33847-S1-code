# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import sqlite3
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
import torch

# 指定本地模型路径
model_path = 'os.environ.get("SENTIMENT_MODEL", "models/distilroberta-finetuned-financial-news-sentiment-analysis")'

# 指定数据库路径
db_path = 'os.environ.get("CORPUS_DB", "data/corpus.db")'

# 指定导出文件路径
output_path = 'os.environ.get("OUT_SENTI", "outputs/senti.csv")'

# 指定标注数据集路径
label_path = 'os.environ.get("LABEL_CSV", "data/label_test.csv")'

# 加载预训练的情感分析模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 假设类别标签如下
class_labels = ["negative", "neutral", "positive"]  # 根据实际模型的输出类别进行调整

# 定义预处理函数
def preprocess_text(text):
    if not isinstance(text, str):
        raise ValueError(f"Input text must be a string, got {type(text)}")

    # 对文本进行分词和编码
    inputs = tokenizer(text=text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    return inputs

# 定义预测函数
def get_prediction(inputs):
    # 获取模型的输出
    outputs = model(**inputs)
    # 获取预测的概率分布
    probabilities = F.softmax(outputs.logits, dim=-1)
    # 获取最高概率的类别索引
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    return predicted_class_id, probabilities

# 连接到数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询数据
page_size = 100  # 每页记录数
offset = 0
all_rows = []

while True:
    cursor.execute("SELECT title FROM my_existing_table LIMIT ? OFFSET ?", (page_size, offset))
    rows = cursor.fetchall()
    if not rows:
        break
    all_rows.extend(rows)
    offset += page_size

# 处理所有新闻标题并存储结果
results = []
for i, title in enumerate(all_rows):
    title = title[0]  # 提取元组中的标题
    if i < 20:
        print(f"Processing title: {title}")
    if title is None:
        print(f"Skipping None title at index {i}")
        continue
    try:
        inputs = preprocess_text(title)
        predicted_class_id, probabilities = get_prediction(inputs)
        predicted_class_label = class_labels[predicted_class_id]
        results.append((title, predicted_class_label, probabilities.tolist()))
    except Exception as e:
        print(f"Error processing title '{title}': {e}")

# 关闭数据库连接
conn.close()

# 写入结果到 CSV 文件
with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Title', 'Predicted Class', 'Probabilities'])
    for result in results:
        writer.writerow(result)

print(f"Results exported to {output_path}")

# 读取标注数据集
true_labels = []
predicted_labels = []

def read_annotated_data(label_path):
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(label_path, 'r', encoding=encoding) as csvfile:
                reader = csv.DictReader(csvfile)
                return list(reader)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file {label_path} with any of the provided encodings")

annotated_data = read_annotated_data(label_path)

# 检查前几条数据
for row in annotated_data[:10]:
    print(row)

# 计算评估指标
for row in annotated_data:
    true_label = row['sentiment']  # 标注数据集的情感标签列名为 sentiment
    title = row['title']  # 标注数据集的标题列名为 title
    try:
        # 对标注数据集的标题进行预处理
        inputs = preprocess_text(title)
        predicted_class_id, _ = get_prediction(inputs)
        predicted_label = class_labels[predicted_class_id]

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    except Exception as e:
        print(f"Error processing annotated title '{title}': {e}")

# 计算评估指标
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Precision: {precision:.4f}")
print(f"Model Recall: {recall:.4f}")
print(f"Model F1 Score: {f1:.4f}")