# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# 设置路径
input_excel_path = r'C:\Users\15610\Desktop\外文.xlsx'  # Excel 文件路径
stop_words_file = r'E:\models\English.txt'  # 英文停用词文件路径
model_dir = r'E:\models\Ceceliachenen\paraphrase-multilingual-MiniLM-L12-v2'  # 多语言模型路径
output_dir = r'C:\Users\15610\Desktop\topic_analysis_results2'  # 输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载英文停用词
with open(stop_words_file, 'r', encoding='utf-8') as file:
    stop_words = set(word.strip().lower() for word in file)


# 加载并预处理Excel数据
def load_and_preprocess_data(file_path, stop_words):
    df = pd.read_excel(file_path)
    df['preprocessed_abstract'] = df['abstract'].apply(
        lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stop_words])
    )
    return df


df = load_and_preprocess_data(input_excel_path, stop_words)

# 初始化多语言BERT模型
model = SentenceTransformer(model_dir)

# 创建HDBSCAN实例，设置聚类参数以减少噪声并控制主题数量
hdbscan_model = HDBSCAN(
    min_cluster_size=20,  # 增大这个值以减少小簇的数量
    min_samples=20,  # 增大这个值以减少噪声点的数量
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# 创建BERTopic实例，指定聚类模型和其他参数
vectorizer_model = CountVectorizer(token_pattern=r'(?u)\b\w{2,}\b')  # 排除单个单词
topic_model = BERTopic(embedding_model=model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model)

# 训练模型
topics, probs = topic_model.fit_transform(df['preprocessed_abstract'].tolist())

# 将主题分配结果添加回原始DataFrame
df['Topic'] = topics

# 统计每个主题的文档数量及噪声数量
topic_counts = df['Topic'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']
print("每个主题的文档数量 (包括噪声):")
print(topic_counts)

# 主题随时间的变化分析 - 按年分组
topic_evolution = df.groupby(['time', 'Topic']).size().unstack(fill_value=0)

# 可视化主题随时间变化 - 折线图
plt.figure(figsize=(14, 7))
for topic in topic_evolution.columns:
    if topic != -1:  # 不绘制噪声主题
        plt.plot(topic_evolution.index, topic_evolution[topic], marker='o', label=f'Topic {topic}')
plt.title('Topic Evolution Over Time (Yearly)')
plt.ylabel('Document Count')
plt.xlabel('Year')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "topic_evolution_over_time_yearly_line.png"))
plt.show()


# 打印关键词及其权重和代表性文档
def print_topic_keywords_weights_and_representative_docs(topic_model, df, top_n_topics=None, n_words=10):
    unique_topics = sorted(df['Topic'].unique())
    if top_n_topics is None:
        top_n_topics = len(unique_topics)  # 默认打印所有主题

    for topic_id in unique_topics[:top_n_topics]:
        keywords = topic_model.get_topic(topic_id)
        if isinstance(keywords, bool) or not keywords:
            keywords = []  # 如果是布尔值或空列表，则设为空列表

        print(f"\nTopic {topic_id}:")
        if keywords:
            print("Keywords and Weights:")
            for word, weight in keywords[:n_words]:
                print(f"  {word}: {weight:.4f}")
        else:
            print("No keywords found for this topic.")

        # 找到属于该主题的文档
        topic_df = df[df['Topic'] == topic_id]
        if not topic_df.empty:
            print("Representative Documents:")
            for idx, row in topic_df.head(3).iterrows():  # 打印每个主题的前3个文档
                print(f"Doc ID {idx}: {row['preprocessed_abstract']}")  # 根据实际情况调整列名
        else:
            print("No documents found for this topic.")


print_topic_keywords_weights_and_representative_docs(topic_model, df)

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

print("主题分析与可视化完成！")