# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
import sqlite3
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import torch
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import hdbscan
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import optuna

# 设置随机种子以确保结果的可重复性
SEED = 42

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK数据路径设置
nltk_data_path = r'os.environ.get("NLTK_DATA", "models/nltk_data")'  # 替换为你的NLTK数据的实际路径
nltk.data.path.append(nltk_data_path)

# 文本预处理函数
def preprocess_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# 加载停用词和词形还原器
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError as e:
    logger.error(f"NLTK data not found at specified path: {e}")
    raise

# 连接数据库并读取数据
db_path = r'os.environ.get("CORPUS_DB", "data/corpus.db")'
try:
    with sqlite3.connect(db_path) as conn:
        query = "SELECT body FROM news"
        df = pd.read_sql_query(query, conn)
except Exception as e:
    logger.error(f"Error connecting to the database or executing query: {e}")
    raise

documents = df['body'].dropna().tolist()
preprocessed_documents = [preprocess_text(doc, lemmatizer, stop_words) for doc in documents]

def objective(trial):
    try:
        n_neighbors = trial.suggest_int('n_neighbors', 5, 30)
        n_components = trial.suggest_int('n_components', 2, 15)
        min_dist = trial.suggest_uniform('min_dist', 0.0, 1.0)
        min_cluster_size = trial.suggest_int('min_cluster_size', 5, 30)
        cluster_selection_epsilon = trial.suggest_uniform('cluster_selection_epsilon', 0.25, 0.8)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_path = r'os.environ.get("EMBEDDING_MODEL", "models/all-MiniLM-L6-v2")'
        embedding_model = SentenceTransformer(model_path).to(device)

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric='cosine',
            random_state=SEED
        )

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
            prediction_data=True,
            cluster_selection_epsilon=cluster_selection_epsilon
        )

        final_model = BERTopic(
            embedding_model=embedding_model,
            top_n_words=15,
            min_topic_size=min_cluster_size,
            calculate_probabilities=False,
            verbose=False,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics="auto"
        )

        topics, probs = final_model.fit_transform(preprocessed_documents)

        valid_topics = [t for t in topics if t != -1]
        if len(set(valid_topics)) < 4:
            raise optuna.exceptions.TrialPruned("Less than 5 topics were generated.")

        umap_embeddings = final_model.umap_model.embedding_
        scaler = StandardScaler()
        scaled_umap_embeddings = scaler.fit_transform(umap_embeddings)

        outlier_indices = [i for i, t in enumerate(topics) if t == -1]
        remaining_outlier_indices = outlier_indices.copy()

        while len(remaining_outlier_indices) > 180:
            topic_centers = {}
            for topic in set(valid_topics):
                topic_embeddings = umap_embeddings[[i for i, t in enumerate(topics) if t == topic]]
                center = np.mean(topic_embeddings, axis=0)
                topic_centers[topic] = center

            new_topics_for_outliers = []
            outlier_embeddings = umap_embeddings[remaining_outlier_indices]
            for outlier_emb in outlier_embeddings:
                distances = [np.linalg.norm(outlier_emb - topic_centers[topic]) for topic in topic_centers]
                assigned_topic = distances.index(min(distances))
                new_topics_for_outliers.append(assigned_topic)

            for idx, new_topic in zip(remaining_outlier_indices, new_topics_for_outliers):
                topics[idx] = new_topic

            remaining_outlier_indices = [i for i, t in enumerate(topics) if t == -1]
            if len(remaining_outlier_indices) <= 180:
                break

        for idx in remaining_outlier_indices:
            topics[idx] = -1

        valid_topics = [t for t in topics if t != -1]
        valid_embeddings = scaled_umap_embeddings[[i for i, t in enumerate(topics) if t != -1]]

        scores = {}
        if len(valid_topics) > 1 and len(valid_embeddings) > 1:
            scores['silhouette'] = silhouette_score(valid_embeddings, valid_topics)
            scores['calinski_harabasz'] = calinski_harabasz_score(valid_embeddings, valid_topics)
        else:
            scores['silhouette'] = 0.0
            scores['calinski_harabasz'] = 0.0

        texts = [[word for word in document.split()] for document in preprocessed_documents]
        id2word = Dictionary(texts)

        topics_info = final_model.get_topic_info()
        topics_list = [final_model.get_topic(topic_id)[:10] for topic_id in topics_info['Topic'] if topic_id != -1]
        topic_words = [[word for word, _ in topic] for topic in topics_list]

        coherence_score = CoherenceModel(topics=topic_words, texts=texts, dictionary=id2word, coherence='c_v',
                                         processes=1).get_coherence()
        scores['coherence'] = coherence_score

        logger.info(f"Silhouette Score: {scores['silhouette']}")
        logger.info(f"Calinski-Harabasz Index: {scores['calinski_harabasz']}")
        logger.info(f"Coherence Score: {scores['coherence']}")

        return scores['silhouette']
    except Exception as e:
        logger.warning(f"Trial failed due to error: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def print_trial_info(trial, final_model, topics):
    logger.info(f"Trial parameters: {trial.params}")
    logger.info(f"Silhouette Score: {trial.value}")

    topics_info = final_model.get_topic_info()
    for index, row in topics_info.iterrows():
        topic_id = row['Topic']
        keywords = ", ".join([word for word, score in final_model.get_topic(topic_id)])
        representative_doc = final_model.get_representative_docs(topic_id)[0]
        logger.info(f"Topic {topic_id}: Keywords: {keywords}, Representative Doc: {representative_doc}")

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    successful_trials = [trial for trial in study.trials if trial.value is not None]
    if len(successful_trials) < 1:
        raise ValueError(f"Not enough successful trials found.")

    for idx, trial in enumerate(successful_trials):
        params = trial.params
        logger.info(f"Running model with parameters from trial {idx}: {params}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_path = r'os.environ.get("EMBEDDING_MODEL", "models/all-MiniLM-L6-v2")'
        embedding_model = SentenceTransformer(model_path).to(device)

        umap_model = UMAP(
            n_neighbors=params['n_neighbors'],
            n_components=params['n_components'],
            min_dist=params['min_dist'],
            metric='cosine',
            random_state=SEED
        )

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True,
            prediction_data=True,
            cluster_selection_epsilon=params['cluster_selection_epsilon']
        )

        final_model = BERTopic(
            embedding_model=embedding_model,
            top_n_words=15,
            min_topic_size=params['min_cluster_size'],
            calculate_probabilities=False,
            verbose=False,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics="auto"
        )

        try:
            topics, probs = final_model.fit_transform(preprocessed_documents)
            # 处理 outliers 和保存结果的逻辑与 objective 函数中相同
            valid_topics = [t for t in topics if t != -1]
            if len(set(valid_topics)) < 4:
                raise optuna.exceptions.TrialPruned("Less than 5 topics were generated.")

            umap_embeddings = final_model.umap_model.embedding_
            scaler = StandardScaler()
            scaled_umap_embeddings = scaler.fit_transform(umap_embeddings)

            outlier_indices = [i for i, t in enumerate(topics) if t == -1]
            remaining_outlier_indices = outlier_indices.copy()

            while len(remaining_outlier_indices) > 180:
                topic_centers = {}
                for topic in set(valid_topics):
                    topic_embeddings = umap_embeddings[[i for i, t in enumerate(topics) if t == topic]]
                    center = np.mean(topic_embeddings, axis=0)
                    topic_centers[topic] = center

                new_topics_for_outliers = []
                outlier_embeddings = umap_embeddings[remaining_outlier_indices]
                for outlier_emb in outlier_embeddings:
                    distances = [np.linalg.norm(outlier_emb - topic_centers[topic]) for topic in topic_centers]
                    assigned_topic = distances.index(min(distances))
                    new_topics_for_outliers.append(assigned_topic)

                for idx, new_topic in zip(remaining_outlier_indices, new_topics_for_outliers):
                    topics[idx] = new_topic

                remaining_outlier_indices = [i for i, t in enumerate(topics) if t == -1]
                if len(remaining_outlier_indices) <= 180:
                    break

            for idx in remaining_outlier_indices:
                topics[idx] = -1
            topics_info = final_model.get_topic_info()
            topic_keywords_weights = []

            for topic_id in topics_info['Topic']:
                if topic_id != -1:  # 排除异常值话题
                    keywords_with_weights = final_model.get_topic(topic_id)  # 获取每个主题的关键词及其权重
                    print(f"Topic {topic_id}:")
                    for kw in keywords_with_weights:
                        keyword, weight = kw
                        print(f"  Keyword: {keyword}, Weight: {weight}")
                    print()

            save_dir = rf"D:\model_results_trial_{trial.number}"
            os.makedirs(save_dir, exist_ok=True)

            final_model.visualize_documents(docs=documents).write_html(os.path.join(save_dir, "documents.html"))
            final_model.visualize_topics().write_html(os.path.join(save_dir, "topics.html"))
            final_model.visualize_barchart().write_html(os.path.join(save_dir, "barchart.html"))
            final_model.visualize_term_rank().write_html(os.path.join(save_dir, "term_rank.html"))

            df_results = pd.DataFrame({'document': documents, 'topic': topics})
            csv_path = os.path.join(save_dir, "model_results.csv")
            df_results.to_csv(csv_path, index=False)

            print_trial_info(trial, final_model, topics)

            logger.info(f"Results for trial {trial.number} saved in directory: {save_dir}")
        except Exception as e:
            logger.warning(f"Failed to run model for trial {idx} due to error: {str(e)}")