from gensim import corpora, models
import jieba.posseg as jp
import jieba

input_content = [line.strip() for line in open ('input.txt', 'r')]
# 老规矩，先分词
words_list = []
for text in input_content:
  words = [w.word for w in jp.cut(text)]
  words_list.append(words)

# 构建文本统计信息, 遍历所有的文本，为每个不重复的单词分配序列id，同时收集该单词出现的次数
dictionary = corpora.Dictionary(words_list)

# 构建语料，将dictionary转化为一个词袋。
# corpus是一个向量的列表，向量的个数就是文档数。你可以输出看一下它内部的结构是怎样的。
corpus = [dictionary.doc2bow(words) for words in words_list]

# 开始训练LDA模型
lda_model = models.ldamodel.LdaModel(corpus=corpus, num_topics=8, id2word=dictionary, passes=10)