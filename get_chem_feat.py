import os
from pathlib import Path
import pickle

import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoModel, RobertaTokenizer


CHOICE = 1

datasets = ['bindingdb', 'biosnap']
dataset_name = datasets[CHOICE]
full = pd.read_csv(f'./datasets/{dataset_name}/fulldata.csv')
unique_drug = full['SMILES'].drop_duplicates().reset_index(drop=True)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu" 
)
chem_bert_path = Path.home() / 'models/ChemBERTa-zinc-base-v1'
chem_embedding_path = Path.home() / f'embedding/ChemBERTa-zinc-base-v1'
os.makedirs(chem_embedding_path, exist_ok=True)

tokenizer = RobertaTokenizer.from_pretrained(chem_bert_path)
model = AutoModel.from_pretrained(chem_bert_path).to(device)
model.eval()

max_length = 512
results = {}

for smiles in tqdm(unique_drug, total=len(unique_drug)):

    inputs = tokenizer(
        smiles, return_tensors="pt",
        max_length=max_length,
        truncation=True, 
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        smiles_feats = outputs.last_hidden_state      
    results[smiles] = smiles_feats.detach().cpu().squeeze()
    

with open(f"{chem_embedding_path}/{dataset_name}_feats.pkl", "wb") as f:
    pickle.dump(results, f)