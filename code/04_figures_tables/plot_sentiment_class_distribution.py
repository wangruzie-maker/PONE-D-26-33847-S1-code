# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np  # 添加这行来导入 numpy

# 指定标注数据集路径（CSV 文件）
csv_path = os.environ.get("OUT_SENTI", "outputs/sentiment_integrated.csv")  # 根据实际情况修改路径

# 定义类别标签
class_labels = ["negative", "neutral", "positive"]


# 尝试多种编码格式读取 CSV 文件
def read_csv_data(csv_path):
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully read CSV file with {encoding} encoding. Total rows: {len(df)}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to read CSV file with {encoding} encoding.")
        except Exception as e:
            print(f"Error reading CSV file with {encoding} encoding: {e}")
            return None
    print("Could not read the CSV file with any of the provided encodings.")
    return None


# 可视化每个类别的预测分布
def plot_class_distribution(predicted_labels):
    label_counts = Counter(predicted_labels)
    labels, counts = zip(*sorted(label_counts.items()))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color='skyblue')

    # 添加红线表示平均值或其他基准值
    if counts:  # 确保 counts 不为空，以防除以零错误
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


# 主函数
if __name__ == "__main__":
    df = read_csv_data(csv_path)

    if df is not None:
        # 假设 CSV 文件包含 'Predicted Class' 列
        # 如果列名不同，请根据实际情况调整
        predicted_labels = df['Predicted Class'].tolist()

        # 确保所有数据都被纳入可视化
        plot_class_distribution(predicted_labels)

        print("All data has been included in the visualizations.")