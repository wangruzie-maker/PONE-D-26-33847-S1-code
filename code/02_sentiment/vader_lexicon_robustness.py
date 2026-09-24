# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import sqlite3
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 下载VADER词典（如果尚未下载）
nltk.download('vader_lexicon')

# 初始化情感分析器
analyzer = SentimentIntensityAnalyzer()

# 定义情感分析函数
def get_sentiment(text):
    if not isinstance(text, str):  # 确保文本是字符串类型
        return 0.0  # 或者其他默认值
    scores = analyzer.polarity_scores(text)
    return scores['compound']  # 返回综合得分

# 定义情感等级分类函数
def classify_sentiment(score):
    if score <= -0.6:
        return -3  # 非常负面
    elif score <= -0.2:
        return -2  # 负面
    elif score < 0:
        return -1  # 稍微负面
    elif score == 0:
        return 0    # 中立
    elif score < 0.6:
        return 1    # 稍微正面
    elif score < 1:
        return 2    # 正面
    else:
        return 3    # 非常正面

# 连接到SQLite数据库
db_path = r'E:\quchong保护0 - 副本.db'  # 替换为你的数据库路径
conn = sqlite3.connect(db_path)

try:
    # 查询所有新闻标题
    query = "SELECT title AS 文本 FROM news"  # 替换为你的表名和列名
    df = pd.read_sql_query(query, conn)

    # 应用情感分析函数，并显示进度条
    df['sentiment'] = [None] * len(df)  # 初始化列以提高性能
    for i, text in tqdm(enumerate(df['文本']), total=len(df), desc="Analyzing sentiments"):
        df.at[i, 'sentiment'] = get_sentiment(text)

    # 添加情感等级列
    df['sentiment_level'] = df['sentiment'].apply(classify_sentiment)

    # 查看结果
    print(df[['文本', 'sentiment', 'sentiment_level']].head())

    # 将结果写回到SQLite数据库
    df.to_sql('news_sentiment', conn, if_exists='replace', index=False)

finally:
    # 关闭数据库连接
    conn.close()

# 设置Seaborn主题
sns.set(style='whitegrid')

# 计算情感等级的计数
sentiment_counts = df['sentiment_level'].value_counts().sort_index()

# 创建条形图
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Level Distribution')
plt.xlabel('Sentiment Level')
plt.ylabel('Count')
plt.xticks(ticks=range(-3, 4), labels=['-3', '-2', '-1', '0', '1', '2', '3'])
plt.show()

# 创建情感得分分布图
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=30, kde=True, color='blue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# 创建饼图
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(sentiment_counts)))
plt.title('Sentiment Level Distribution')
plt.axis('equal')  # 使饼图为圆形
plt.show()