import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.data import Field, Example, Dataset, BucketIterator

import pandas as pd
from tqdm.auto import tqdm


from tools.constants import *


def prepare_data():
    import pickle

    word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    fields = [('source', word_field), ('target', word_field)]


    data = pd.read_csv(NEWS_PATH, delimiter=',')

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))



    dataset = Dataset(examples, fields)

    train_dataset, test_dataset = dataset.split(split_ratio=0.85)

    print('Train size =', len(train_dataset))
    print('Test size =', len(test_dataset))

    word_field.build_vocab(train_dataset, min_freq=7)
    print('Vocab size =', len(word_field.vocab))

    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=get_device(), sort=False
    )

    print('Finished preparing dataset')



    with open(DATASET_PATH, 'wb') as f:
        pickle.dump({
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'word_field': word_field
        }, f, protocol=pickle.HIGHEST_PROTOCOL)