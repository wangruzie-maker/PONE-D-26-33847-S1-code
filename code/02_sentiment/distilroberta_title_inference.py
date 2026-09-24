# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import sqlite3
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np  # 确保导入 NumPy

# 指定本地模型路径
model_path = os.environ.get("SENTIMENT_MODEL", "models/distilroberta-finetuned-financial-news-sentiment-analysis")

# 指定数据库路径
db_path = os.environ.get("CORPUS_DB", "data/corpus.db")

# 指定导出文件路径
output_path = os.environ.get("OUT_SENTI", "outputs/sentiment_integrated.csv")

# 指定标注数据集路径
label_path = os.environ.get("LABEL_CSV", "data/label_test.csv")

# 加载预训练的情感分析模型（三级分类）
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 定义类别标签
class_labels = ["negative", "neutral", "positive"]


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
    probabilities = F.softmax(outputs.logits, dim=-1).squeeze()
    # 获取最高概率的类别索引
    predicted_class_id = torch.argmax(probabilities).item()
    return predicted_class_id, probabilities.tolist()


# 连接到数据库并查询数据
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取总记录数，以便确认是否处理了所有数据
cursor.execute("SELECT COUNT(*) FROM wpnews")
total_records = cursor.fetchone()[0]
print(f"Total records in the database: {total_records}")

# 查询数据
page_size = 100  # 每页记录数
offset = 0
all_rows = []

processed_records = 0
while processed_records < total_records:
    cursor.execute("SELECT title FROM wpnews LIMIT ? OFFSET ?", (page_size, offset))
    rows = cursor.fetchall()
    if not rows:
        break
    all_rows.extend(rows)
    processed_records += len(rows)
    offset += page_size
    print(f"Processed {processed_records} out of {total_records} records.")

# 确保处理了所有记录
if processed_records != total_records:
    print("Warning: Not all records were processed.")

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
        results.append((title, predicted_class_label, probabilities))
    except Exception as e:
        print(f"Error processing title '{title}': {e}")

# 关闭数据库连接
conn.close()

# 写入结果到 CSV 文件
with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Title', 'Predicted Class', 'Probabilities'])
    for result in results:
        writer.writerow([result[0], result[1], ','.join(map(str, result[2]))])

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
                data = list(reader)
                print(f"Total annotated titles read from file: {len(data)}")  # 打印读取的数据量
                return data
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file {label_path} with any of the provided encodings")


annotated_data = read_annotated_data(label_path)

# 计算评估指标
for row in annotated_data:
    true_label = row.get('sentiment')  # 标注数据集的情感标签列名为 sentiment
    title = row.get('title')  # 标注数据集的标题列名为 title
    if true_label is None or title is None:
        print(f"Skipping row due to missing data: {row}")
        continue
    try:
        # 对标注数据集的标题进行预处理
        inputs = preprocess_text(title)
        predicted_class_id, _ = get_prediction(inputs)
        predicted_label = class_labels[predicted_class_id]

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    except Exception as e:
        print(f"Error processing annotated title '{title}': {e}")

# 确保所有标注数据都被处理
print(f"Total annotated titles processed: {len(annotated_data)}")
print(f"Total true labels collected: {len(true_labels)}")
print(f"Total predicted labels collected: {len(predicted_labels)}")

# 如果有缺失的数据，找出原因
if len(annotated_data) != len(true_labels) or len(annotated_data) != len(predicted_labels):
    print("Warning: Some annotated data was skipped or failed to process.")
    print(
        f"Expected {len(annotated_data)}, but got {len(true_labels)} true labels and {len(predicted_labels)} predicted labels.")

# 计算评估指标
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Precision: {precision:.4f}")
print(f"Model Recall: {recall:.4f}")
print(f"Model F1 Score: {f1:.4f}")


# 可视化混淆矩阵
def plot_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap=plt.cm.Blues)

    # 添加对角线上的红线以突出正确分类的情况
    for i in range(cm.shape[0]):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.title('Confusion Matrix with Highlighted Correct Classifications')
    plt.show()


# 可视化每个类别的预测分布
def plot_class_distribution(predicted_labels):
    label_counts = Counter(predicted_labels)
    labels, counts = zip(*sorted(label_counts.items()))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color='skyblue')

    # 添加红线表示平均值或其他基准值
    avg_count = np.mean(counts)
    ax.axhline(y=avg_count, color='r', linestyle='--', label=f'Average: {avg_count:.2f}')

    # 添加数据标签到柱状图
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Classes')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 调用绘图函数前，确保所有数据都被包含
plot_confusion_matrix(true_labels, predicted_labels)
plot_class_distribution(predicted_labels)

# 确认所有数据都被纳入可视化
print("All data has been included in the visualizations.")