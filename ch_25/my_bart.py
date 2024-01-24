import torch
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import datasets
# from transformers.modeling_bart import shift_tokens_right
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.models.bart.modeling_bart import shift_tokens_right

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

dataset = load_dataset("cnn_dailymail", "3.0.0")
train_dataset = dataset["train"]
test_dataset = dataset["test"]


def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['article'], pad_to_max_length=True, max_length=1024,
                                                  truncation=True)
    target_encodings = tokenizer.batch_encode_plus(example_batch['highlights'], pad_to_max_length=True, max_length=1024,
                                                   truncation=True)

    labels = torch.tensor(target_encodings['input_ids'])
    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    labels[labels[:, :] == model.config.pad_token_id] = -100

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    }
    return encodings


dataset = dataset.map(convert_to_features, batched=True)
columns = ['input_ids', 'labels', 'decoder_input_ids', 'attention_mask', ]
dataset.set_format(type='torch', columns=columns)

training_args = Seq2SeqTrainingArguments(
    output_dir='./models/bart-summarizer',  # 模型输出目录
    num_train_epochs=1,  # 训练轮数
    per_device_train_batch_size=1,  # 训练过程bach_size
    per_device_eval_batch_size=1,  # 评估过程bach_size
    warmup_steps=500,  # 学习率相关参数
    weight_decay=0.01,  # 学习率相关参数
    logging_dir='./logs',  # 日志目录
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

trainer.train()
