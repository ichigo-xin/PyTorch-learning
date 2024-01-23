import torchtext

train_iter = torchtext.datasets.IMDB(root='./data', split='train')
train_iter = iter(train_iter)  # Convert to an iterator

# 创建分词器
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')


# 构建词汇表
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 数据处理pipelines
text_pipeline = lambda x: vocab(tokenizer(x))
# label_pipeline = lambda x: 1 if x == 'pos' else 0

print(text_pipeline('here is the an example'))
