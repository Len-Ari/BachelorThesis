import os
import torch

# SentenceTransformer libary
# conda install -c conda-forge sentence-transformers
from sentence_transformers import SentenceTransformer

def save_model_sentencetransformer(model_name, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model = SentenceTransformer(model_name)
    model.save_pretrained(path=save_dir)

def load_model_sentencetransformer(save_dir, usecpu=False):
    assert os.path.isdir(save_dir)
    if torch.cuda.is_available() and not usecpu:
        device = "cuda"
    else:
        device = "cpu"
    model = SentenceTransformer(save_dir, device=torch.device(device))
    return model