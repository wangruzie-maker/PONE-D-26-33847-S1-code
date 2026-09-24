# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 定义类别标签及其对应的数值
class_values = {"negative": -1, "neutral": 0, "positive": 1}

# 指定标注数据集路径（CSV 文件）
csv_path = 'os.environ.get("OUT_SENTI", "outputs/sentiment_integrated.csv")'  # 修改为您的 CSV 文件绝对路径

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

# 计算情感评分并按月份聚合
def calculate_sentiment_score_by_month(df):
    # 确保日期列存在且为 datetime 格式
    if 'Date' not in df.columns or 'Predicted Class' not in df.columns:
        raise ValueError("CSV file must contain 'Date' and 'Predicted Class' columns.")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # 删除无效日期行

    # 添加情感评分列
    df['Sentiment Score'] = df['Predicted Class'].map(class_values)

    # 按月份分组并计算平均情感评分
    df['Month'] = df['Date'].dt.to_period('M')  # 使用 Period 类型表示月份
    monthly_avg_scores = df.groupby('Month')['Sentiment Score'].mean().reset_index()

    # 将月份转换回 datetime 类型以便绘图
    monthly_avg_scores['Month'] = monthly_avg_scores['Month'].dt.to_timestamp()

    return monthly_avg_scores

# 绘制情感波动时序图
def plot_sentiment_timeline(monthly_avg_scores):
    plt.figure(figsize=(14, 7))
    plt.plot_date(monthly_avg_scores['Month'], monthly_avg_scores['Sentiment Score'], linestyle='-', marker=None)
    plt.title('Average Sentiment Score Over Months')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.tight_layout()  # 调整布局以防止标签被裁剪
    plt.show()

# 主函数
if __name__ == "__main__":
    df = read_csv_data(csv_path)

    if df is not None:
        # 假设 CSV 文件包含 'Date' 和 'Predicted Class' 列
        monthly_avg_scores = calculate_sentiment_score_by_month(df)

        # 绘制情感波动时序图
        plot_sentiment_timeline(monthly_avg_scores)

        print("All data has been included in the visualizations.")