# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import pandas as pd
import jieba
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


# 加载 Excel 数据
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df


# 预处理文本（中文分词和去除停用词）
def preprocess_text(text, stop_words):
    tokens = jieba.lcut(text)
    filtered_tokens = [token for token in tokens if token.strip() and token not in stop_words and len(token) > 1]
    return " ".join(filtered_tokens)


# 定义 tokenizer 函数以替代 lambda 表达式
def custom_tokenizer(text):
    return text.split()


# 打印关键词及其频数
def print_keyword_frequencies(topic_model, preprocessed_abstracts):
    all_tokens = []
    for doc in preprocessed_abstracts:
        tokens = doc.split()
        all_tokens.extend(tokens)

    freq_dist = Counter(all_tokens)

    for topic_num in topic_model.get_topic_info().Topic.unique():
        keywords = topic_model.get_topic(topic_num)
        print(f"Topic {topic_num}:")
        for word, weight in keywords:
            print(f"  {word}: {freq_dist[word]} (weight: {weight:.4f})")
        print()


# 列出每个主题的三条代表性文档
def print_representative_docs(topic_model, df, top_n=3):
    for topic_num in sorted(df['Topic'].unique()):
        if topic_num == -1:
            continue
        subset = df[df['Topic'] == topic_num].copy()
        subset.sort_values(by='time', inplace=True)
        representative_docs = subset.head(top_n)
        print(f"Topic {topic_num} Representative Documents:")
        for i, (_, row) in enumerate(representative_docs.iterrows(), start=1):
            print(f"  Doc {i} ({row['time'].year}): {row['abstract'][:200]}...")  # 打印文档的前200个字符
        print()


# 设置路径
input_excel_path = r'C:\Users\15610\Desktop\中文.xlsx'  # 替换为你的Excel文件路径
stop_words_file = r'E:\models\chinese.txt'  # 替换为你的中文停用词文件路径
model_dir = r'E:\models\Ceceliachenen\paraphrase-multilingual-MiniLM-L12-v2'  # 替换为你的模型路径
output_dir = r'C:\Users\15610\Desktop\topic_analysis_results'  # 输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载中文停用词
with open(stop_words_file, 'r', encoding='utf-8') as file:
    stop_words = set(word.strip() for word in file)

# 加载 Excel 数据
df = load_excel_data(input_excel_path)

# 检查是否有足够的文档
if len(df) == 0:
    raise ValueError("没有加载到任何文档，请检查输入文件路径和格式。")

# 假设 Excel 文件中有两列：'time' 和 'abstract'
df['preprocessed_abstract'] = df['abstract'].apply(lambda x: preprocess_text(x, stop_words))
df['time'] = pd.to_datetime(df['time'])

# 初始化BERT模型，使用本地路径加载模型
model_zh = SentenceTransformer(model_dir)

# 创建HDBSCAN实例，设置聚类参数以减少噪声并控制主题数量
hdbscan_model = HDBSCAN(
    min_cluster_size=15,  # 调整此值以影响主题数量
    min_samples=2,
    metric='euclidean',
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.5
)

# 创建BERTopic实例，指定聚类模型和其他参数
vectorizer_model = CountVectorizer(tokenizer=custom_tokenizer, token_pattern=r'(?u)\b\w{2,}\b')  # 排除单个汉字
topic_model = BERTopic(embedding_model=model_zh, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)

# 训练模型
topics, probs = topic_model.fit_transform(df['preprocessed_abstract'].tolist())

# 将主题分配结果添加回原始DataFrame
df['Topic'] = topics

# 统计每个主题的文档数量及噪声数量
topic_counts = pd.Series(topics).value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']
noise_count = topic_counts.loc[topic_counts['Topic'] == -1, 'Count'].values[0] if -1 in topic_counts[
    'Topic'].values else 0
print(f"噪声（未分类文档）数量: {noise_count}")
print("每个主题的文档数量:")
print(topic_counts[topic_counts['Topic'] != -1])  # 排除噪声主题

# 确保至少有五个主题且每个主题至少有10篇文档
while len(topic_counts[topic_counts['Topic'] >= 0]) < 5 or any(topic_counts[topic_counts['Topic'] >= 0]['Count'] < 10):
    print("主题数量不足或有主题文档数少于10，尝试调整聚类参数...")
    # 尝试减小min_cluster_size或其他参数调整
    hdbscan_model.min_cluster_size -= 1

    # 重新训练模型
    topics, probs = topic_model.fit_transform(df['preprocessed_abstract'].tolist())
    df['Topic'] = topics

    # 更新主题计数
    topic_counts = pd.Series(topics).value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']

    # 打印当前主题数量和文档数量
    print(f"当前主题数量: {len(topic_counts[topic_counts['Topic'] >= 0])}")
    print("每个主题的文档数量:")
    print(topic_counts[topic_counts['Topic'] != -1])

    noise_count = topic_counts.loc[topic_counts['Topic'] == -1, 'Count'].values[0] if -1 in topic_counts[
        'Topic'].values else 0
    print(f"噪声（未分类文档）数量: {noise_count}")

# 主题随时间的变化分析 - 按年分组
df['Year'] = df['time'].dt.year
topic_evolution = df.groupby(['Year', 'Topic']).size().unstack(fill_value=0).T

# 可视化主题随时间变化 - 折线图
plt.figure(figsize=(14, 7))
for topic in topic_evolution.index:
    if topic == -1:
        continue
    plt.plot(topic_evolution.loc[topic], marker='o', label=f'Topic {topic}')
plt.title('Topic Evolution Over Time (Yearly)')
plt.ylabel('Document Count')
plt.xlabel('Year')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "topic_evolution_over_time_yearly_line.png"))
plt.show()

# 其余可视化部分（如主题分布、关键词图等）
try:
    # 主题分布
    topic_model.visualize_topics().write_html(os.path.join(output_dir, "topics_distribution.html"))

    # 主题分层
    topic_model.visualize_hierarchy().write_html(os.path.join(output_dir, "topics_hierarchy.html"))

    # 关键词图
    topic_model.visualize_barchart(top_n_topics=10, n_words=10).write_html(
        os.path.join(output_dir, "keywords_barchart.html"))

    # 主题相似度矩阵
    topic_model.visualize_heatmap().write_html(os.path.join(output_dir, "topics_similarity_heatmap.html"))
except Exception as e:
    print(f"可视化过程中遇到错误: {e}")

# 打印关键词及其频数
print_keyword_frequencies(topic_model, df['preprocessed_abstract'].tolist())

# 列出每个主题的三条代表性文档
print_representative_docs(topic_model, df)

print("主题分析与可视化完成！")