# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import pandas as pd
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import multiprocessing as mp
# 使用绝对路径打开文件
file_path = 'E:/data456.xlsx'

# 更详细的预处理函数
def preprocess(text):
    if isinstance(text, str):
        # 去除标点符号和数字
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # 转换为小写
        text = text.lower()
        # 分词
        tokens = nltk.word_tokenize(text)
        # 去除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # 词形还原
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # 去除低频词
        freq_dist = nltk.FreqDist(tokens)
        tokens = [word for word in tokens if freq_dist[word] > 1]
        return tokens
    return []

def main():
    # 读取文件
    df = pd.read_excel(file_path)

    # 假设文本位于第一列，提取该列的数据为文本列表
    documents = df.iloc[:, 2].tolist()

    # 预处理文档
    processed_docs = [preprocess(doc) for doc in documents]

    # 创建字典和语料库
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # 训练LDA模型
    lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=30, alpha='auto', eta='auto')

    # 打印每个主题及其关键词
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")

    # 计算一致性得分
    coherence_model = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score}")

if __name__ == '__main__':
    mp.freeze_support()
    main()