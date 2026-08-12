import os
from pathlib import Path
import re
import pickle

import pandas as pd
from tqdm import tqdm

import torch
from transformers import EsmModel, EsmTokenizer


CHOICE = 1

datasets = ['bindingdb', 'biosnap']
dataset_name = datasets[CHOICE]
full = pd.read_csv(f'./datasets/{dataset_name}/fulldata.csv')
unique_protein = full['Protein'].drop_duplicates().reset_index(drop=True)

def preprocess_seq(seq):
    seq = re.sub(r"[UZOB]", "X", seq)
    return ' '.join(list(seq))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu" 
)
esm_model_path = Path.home() / 'models/esm2_t33_650M_UR50D'
esm_embedding_path = Path.home() / f'embedding/esm2_t33_650M_UR50D'
os.makedirs(esm_embedding_path, exist_ok=True)

tokenizer = EsmTokenizer.from_pretrained(esm_model_path, do_lower_case=False)
model = EsmModel.from_pretrained(esm_model_path).to(device)
model.eval()


max_length = 1026
results = {}
for seq in tqdm(unique_protein, total=len(unique_protein)):
    inputs = tokenizer(
        preprocess_seq(seq), return_tensors="pt",
        max_length=max_length,
        truncation=True, 
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        seq_feats = outputs.last_hidden_state
    results[seq] = seq_feats.detach().cpu().squeeze()

with open(f"{esm_embedding_path}/{dataset_name}_feats.pkl", "wb") as f:
    pickle.dump(results, f)