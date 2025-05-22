


BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
NEWS_PATH = 'data/news.csv'
DATASET_PATH = 'backups/token_dataset.pkl'
EMB_PATH = 'data/cc.ru.300.bin' # стоит это загрузить с https://fasttext.cc/docs/en/crawl-vectors.html
MODEL_PATH = 'backups/model.pt'


def get_device():
    import torch
    if torch.cuda.is_available():
        from torch.cuda import FloatTensor, LongTensor
        return torch.device('cuda')
    else:
        from torch import FloatTensor, LongTensor
        return torch.device('cpu')