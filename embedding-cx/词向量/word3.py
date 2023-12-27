with open('/Users/chenxin/近期工作/Embedding算法/text8.txt', 'r', encoding='utf-8') as file:
    text = file.read()

from nltk.tokenize import word_tokenize
import string
import nltk
# nltk.download('punkt')  # 下载必要的分词器

# 分词并去除标点符号
tokens = word_tokenize(text.lower())
words = [word for word in tokens if word.isalpha()]

from gensim.models import Word2Vec

# 训练Word2Vec模型
vecsize = 15
model = Word2Vec(sentences=[words], vector_size=vecsize , window=5, min_count=1, workers=4)

word_vectors = model.wv

# 获取单词的向量表示
vector = word_vectors['example']

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 选择两个单词
word1 = 'france'
word2 = 'paris'

# 获取这两个单词的向量表示
vector1 = model.wv[word1]
vector2 = model.wv[word2]
#vector3 = model.wv[word3]

# 计算两个向量的余弦相似度
similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
print(similarity)

# 选择两个单词
word1 = 'king'
word2 = 'queen'
#word3 = 'queen'

# 获取这两个单词的向量表示
vector1 = model.wv[word1]
vector2 = model.wv[word2]
#vector3 = model.wv[word3]

vector11 = vector1 - vector2
# 计算两个向量的余弦相似度
similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
print(similarity)

# 选择两个单词
word1 = 'man'
word2 = 'woman'
#word3 = 'queen'

# 获取这两个单词的向量表示
vector1 = model.wv[word1]
vector2 = model.wv[word2]
#vector3 = model.wv[word3]

# 计算两个向量的余弦相似度
similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
print(similarity)
vector22 = vector1 - vector2

# 假设 vector1 和 vector2 是两个向量，matrix_shape 是矩阵的形状
matrix_shape = (2, vecsize )

# 将两个向量组合成一个矩阵
#np_array = np.reshape([vector1, vector2,vector3], matrix_shape)
np_array = np.reshape([vector11, vector22], matrix_shape)

import pandas as pd
import numpy as np
import openpyxl

# 将矩阵数据转换为 pandas 的 DataFrame
df = pd.DataFrame(np_array)

# 保存为 Excel 文件
df.to_excel('matrix_data2.xlsx', index=False)  # 文件名为 matrix_data.xlsx，不包含行索引
