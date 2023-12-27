from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

#nltk.download('punkt')  # 下载必要的分词器

# 示例文本
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# 对文本进行分词处理
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# 训练Word2Vec模型
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# 获取单词的向量
word_vectors = model.wv

# 获取单词“document”的向量表示
vector = word_vectors['document']

# 找到与给定单词最相似的单词
similar_words = model.wv.most_similar('document')

# 输出结果
print("Vector representation of 'document':", vector)
print("Words most similar to 'document':", similar_words)
