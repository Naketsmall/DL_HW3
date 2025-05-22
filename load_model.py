import torch

from tools.constants import MODEL_PATH
from model_sources.transformer_functs import *
from model_sources.transformer import *

model = torch.load(MODEL_PATH)
model.eval()

"""
push your test val here
"""

val_loss = do_epoch(model, criterion, val_iter, None, name_prefix + '  Val:')