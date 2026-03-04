import random
import torch.nn as nn
import pickle
import os
import torch
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import multiprocessing
from math import ceil
from MolecularUtils import ModificationUtils
import tempfile
import shutil
import json

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MSDataset(nn.Module):
    def __init__(self, args, type: str, data_info: dict):
        super().__init__()
        # parameter
        self.device = args.get_config('General', 'device')

        if args.get_config('General', 'dtype', default='float') == 'float':
            self.dtype = torch.float32
        else:
            self.dtype = torch.half
        
        self.padding_token = args.get_config('Database', 'padding_token', default=31)
        self.max_workers = args.get_config('General', 'threads', default=1)

        if type == 'train':
            self.data = data_info['train_data']
            self.data_length = len(self.data)
        elif type == 'val':
            self.data = data_info['val_data']
            self.data_length = len(self.data)
        elif type == 'test':
            self.data = data_info['test_data']
            self.data_length = len(self.data)
        else:
            raise ValueError(f"Invalid dataset type: {type}")

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        sample = self.data[index]
        
        return {"precursor_chrom": torch.tensor(sample['precursor_chrom'], dtype=self.dtype),
                "precursor_ppm": torch.tensor(sample['precursor_ppm'], dtype=self.dtype),
                "precursor_mz": torch.tensor(sample['precursor_mz'], dtype=self.dtype),
                "precursor_RT": torch.tensor(sample['precursor_RT'], dtype=self.dtype),
                "peptide_ids": torch.tensor(sample['peptide_ids'], dtype=torch.int),
                "modification_ids": torch.tensor(sample['modification_ids'], dtype=torch.int),
                "charge": torch.tensor(sample['charge'], dtype=torch.int),
                "fragment_chrom": torch.tensor(sample['fragment_chrom'], dtype=self.dtype),
                "fragment_ppm": torch.tensor(sample['fragment_ppm'], dtype=self.dtype),
                "fragment_mz": torch.tensor(sample['fragment_mz'], dtype=self.dtype),
                "fragment_RT": torch.tensor(sample['fragment_RT'], dtype=self.dtype),
                "label": torch.tensor(sample['label'], dtype=self.dtype),
                "modified_peptide": sample['modified_peptide']
                }

    def collect_fn(self, batch):
        max_seq_len = max([item["peptide_ids"].shape[0] for item in batch])
        peptide_ids = []
        modification_ids = []
        for item in batch:
            padding_length = max_seq_len - len(item["peptide_ids"])
            if padding_length > 0:
                padded_peptide_ids = torch.cat([item["peptide_ids"], torch.full((padding_length,), self.padding_token, dtype=torch.int)])
                padded_modification_ids = torch.cat([item["modification_ids"], torch.zeros(padding_length, dtype=torch.int)])
            else:
                padded_peptide_ids = item["peptide_ids"]
                padded_modification_ids = item["modification_ids"]
            peptide_ids.append(padded_peptide_ids)
            modification_ids.append(padded_modification_ids)

        return {
            "precursor_chrom": torch.stack([item["precursor_chrom"] for item in batch]).to(self.device),
            "precursor_ppm": torch.stack([item["precursor_ppm"] for item in batch]).to(self.device),
            "precursor_mz": torch.stack([item["precursor_mz"] for item in batch]).to(self.device),
            "precursor_RT": torch.stack([item["precursor_RT"] for item in batch]).to(self.device),
            "peptide_ids": torch.stack(peptide_ids).to(self.device),
            "modification_ids": torch.stack(modification_ids).to(self.device),
            "charge": torch.stack([item["charge"] for item in batch]).to(self.device),
            "fragment_chrom": torch.stack([item["fragment_chrom"] for item in batch]).to(self.device),
            "fragment_ppm": torch.stack([item["fragment_ppm"] for item in batch]).to(self.device),
            "fragment_mz": torch.stack([item["fragment_mz"] for item in batch]).to(self.device),
            "fragment_RT": torch.stack([item["fragment_RT"] for item in batch]).to(self.device),
            "label": torch.stack([item["label"] for item in batch]).to(self.device),
            "modified_peptide": [item["modified_peptide"] for item in batch],
        }

if __name__ == "__main__":
    pass
