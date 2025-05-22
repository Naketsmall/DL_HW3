import os
import pickle


from tools.constants import *
from tools import tokenizer

from model_sources.transformer import *
from model_sources.transformer_functs import *


if not os.path.exists(DATASET_PATH):
    tokenizer.prepare_data()
elif os.path.getsize(DATASET_PATH) == 0:
    tokenizer.prepare_data()

with open(DATASET_PATH, 'rb') as f:
    
    data = pickle.load(f)
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    word_field = data['word_field']

print('Tokenized dataset loadded')


ft_model = fasttext.load_model(bin_path)
embedding_dim = ft_model.get_dimension()


pad_idx = word_field.vocab.stoi['<pad>']

model = EncoderDecoder(word_field.vocab, ft_model, source_vocab_size=len(word_field.vocab)).to(get_device())

criterion = LabelSmoothingLoss(ignore_index=pad_idx).to(get_device())

optimizer = NoamOpt(model, model.d_model)
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

train_iter, test_iter = BucketIterator.splits(
    datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False
)

fit(model, criterion, optimizer, train_iter, epochs_count=30, val_iter=test_iter)


news_text = """Автомобильный аккумулятор может разрядиться по разным причинам: неисправности, связанные с утечкой тока, ресурс батареи исчерпан, оплошность водителя (не выключил фары). Не каждую машину можно завести с толкача и искать автовладельца, готового “прикурить” от своей АКБ, не всегда есть время. Надежнее для таких случаев иметь собственное пусковое или пуско-зарядное устройство"""

print(generate_summary(news_text, model, word_field))
