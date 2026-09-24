# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import sqlite3
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import logging
import pandas as pd
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from sentence_transformers import SentenceTransformer
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置路径和其他配置
db_path = r'os.environ.get("CORPUS_DB", "data/corpus.db")'
table_name = 'wpnews'
evaluation_results_path = 'os.environ.get("OUT_EVAL", "outputs/topic_eval.csv")'
output_file_path = 'os.environ.get("OUT_TOPICS", "outputs/topics.csv")'
nltk_data_path = r'os.environ.get("NLTK_DATA", "models/nltk_data")'
sentence_bert_model_path = r'os.environ.get("EMBEDDING_MODEL", "models/all-MiniLM-L6-v2")'
visualizations_output_dir = 'os.environ.get("OUT_VIZ", "outputs/visualizations")'  # 新增：用于保存可视化的目录

# 确保 NLTK 使用自定义路径
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)


# 加载资源并初始化模型
def load_resources():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # 确保GPU可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using {device} for embedding model")

    embedding_model = SentenceTransformer(sentence_bert_model_path, device=device)
    return stop_words, lemmatizer, embedding_model


# 加载数据库中的数据并预处理
def load_and_preprocess_data(db_path, table_name, stop_words, lemmatizer):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT body, date FROM {table_name}", conn)
    conn.close()

    texts = df.body.tolist()
    dates = df.date.tolist()
    processed_texts = []
    qtar = tqdm(total=len(texts), desc="Processing texts")
    for text in texts:
        qtar.update(1)
        # 分词、去停用词、去标点符号、词形还原
        words = word_tokenize(text.lower())
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        processed_text = ' '.join(filtered_words)
        processed_texts.append(processed_text)
    qtar.close()
    logging.info(f"Processed {len(processed_texts)} texts")
    return processed_texts, dates


# 计算评价指标
def evaluate_topics(processed_texts, topics, probabilities, model):
    ch_score = calinski_harabasz_score(probabilities, topics) if len(set(topics)) > 1 else 0
    silhouette_avg = silhouette_score(probabilities, topics, metric='cosine') if len(set(topics)) > 1 else 0

    # 准备用于一致性分数计算的词汇表和文本
    texts = [doc.split() for doc in processed_texts]
    dictionary = Dictionary(texts)

    topic_keywords_list = []
    for topic in set(topics):
        if topic != -1:  # Skip noise/outliers
            keywords = model.get_topic(topic)
            keyword_ids = [dictionary.token2id[word] for word, _ in keywords if word in dictionary.token2id]
            if keyword_ids:
                topic_keywords_list.append(keyword_ids)

    coherence_score = 0
    if topic_keywords_list:
        coherence_model = CoherenceModel(
            topics=topic_keywords_list,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

    return ch_score, silhouette_avg, coherence_score


# 保存主题信息到 CSV 文件
def save_topics_to_csv(model, processed_texts, output_file_path):
    topic_info_df = model.get_topic_info()
    topic_info = []

    for index, topic in topic_info_df.iterrows():
        if topic['Count'] == 0 or topic['Topic'] == -1:
            continue

        top_keywords = model.get_topic(topic['Topic'])[:10]
        top_keywords_str = ', '.join([f"{word} ({score:.4f})" for word, score in top_keywords])

        topic_docs_indices = [i for i, t in enumerate(model.topics_) if t == topic['Topic']]
        representative_doc = processed_texts[topic_docs_indices[0]] if topic_docs_indices else "No documents"

        topic_info.append({
            'Topic': topic['Topic'],
            'Keywords': top_keywords_str,
            'Representative_Document': representative_doc,
            'Num_Documents': topic['Count']
        })

    topic_df = pd.DataFrame(topic_info)
    topic_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    logging.info(f"Topics information saved to {output_file_path}")


# 可视化函数
def visualize_topics(model, processed_texts, dates, visualizations_output_dir):
    os.makedirs(visualizations_output_dir, exist_ok=True)

    visualization_functions = [
        ('topics_distribution.html', lambda: model.visualize_topics()),
        ('topics_hierarchy.html', lambda: model.visualize_hierarchy()),
        ('heatmap.html', lambda: model.visualize_heatmap()),
        ('topics_similarity.html', lambda: model.visualize_barchart())
    ]

    # 生成除主题演进图外的其他可视化
    for filename, func in visualization_functions:
        try:
            fig = func()
            fig.write_html(os.path.join(visualizations_output_dir, filename))
            logging.info(f"{filename.capitalize()} visualization saved.")
        except Exception as e:
            logging.error(f"Failed to generate {filename}: {e}")
            continue

    # 特别处理主题演进图
    try:
        topics_over_time = model.topics_over_time(processed_texts, timestamps=dates, nr_bins=50)
        evolution_fig = model.visualize_topics_over_time(topics_over_time)
        evolution_fig.update_layout(title="Topics Over Time")
        evolution_fig.write_html(os.path.join(visualizations_output_dir, "evolution.html"))
        logging.info("Evolution visualization saved.")
    except Exception as e:
        logging.warning(f"Failed to generate topics over time visualization: {e}")

    logging.info("All available visualizations have been generated.")


# 主函数：聚类和评估
def main_clustering_and_evaluation(num_topics):
    stop_words, lemmatizer, embedding_model = load_resources()
    processed_texts, dates = load_and_preprocess_data(db_path, table_name, stop_words, lemmatizer)

    print('# 3. 训练模型')
    model = BERTopic(embedding_model=embedding_model, calculate_probabilities=True, top_n_words=10, nr_topics=num_topics)
    topics, probabilities = model.fit_transform(processed_texts)

    # 打印前十个关键词
    for topic_id in set(topics):
        if topic_id != -1:
            keywords = model.get_topic(topic_id)[:10]
            print(f"Topic {topic_id}: {keywords}")

    # 执行评估
    ch_score, silhouette_avg, coherence_score = evaluate_topics(processed_texts, topics, probabilities, model)
    logging.info(
        f"Evaluated metrics for {num_topics} topics: CH={ch_score}, Silhouette={silhouette_avg}, Coherence={coherence_score}"
    )

    # 保存评估结果到 CSV 文件
    results_df = pd.DataFrame([[num_topics, ch_score, silhouette_avg, coherence_score]],
                              columns=['num_topics', 'ch_score', 'silhouette_score', 'coherence_score'])
    results_df.to_csv(evaluation_results_path, mode='a', header=not os.path.exists(evaluation_results_path),
                      index=False, encoding='utf-8-sig')
    logging.info(f"Evaluation results appended to {evaluation_results_path}")

    # 保存每个主题的前十个关键词和代表文档到 CSV 文件
    save_topics_to_csv(model, processed_texts, output_file_path)

    # 生成可视化
    visualize_topics(model, processed_texts, dates, visualizations_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a BERTopic model and evaluate it.")
    parser.add_argument('--num_topics', type=int, default=12, help="Number of topics to train the model with.")
    args = parser.parse_args()

    main_clustering_and_evaluation(args.num_topics)