import math

from nltk import word_tokenize
from nltk import TextCollection

sents = ['i like jike', 'i want to eat apple', 'i like lady gaga']
# 首先进行分词
sents = [word_tokenize(sent) for sent in sents]

# 构建语料库
corpus = TextCollection(sents)

# 计算TF
tf = corpus.tf('like', corpus)

# 计算IDF
idf = corpus.idf('lik')

# 计算任意一个单词的TF-IDF
tf_idf = corpus.tf_idf('like', corpus)

print(tf)
print(idf)
print(tf_idf)
print(math.log(3 / 2))