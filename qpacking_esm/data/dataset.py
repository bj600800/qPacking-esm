"""
------------------------------------------------------------------------------
Author:    Dou Zhixin
Email:     bj600800@gmail.com
Date:      2025/06/05

Description:
    Dataset construction utilities for qPacking / ESM fine-tuning
------------------------------------------------------------------------------
"""

import os
import json
import pickle
import random
import numpy as np
from Bio import SeqIO

import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_from_disk

from qpacking_esm.common.analyze_feature import load_existing_results
from qpacking_esm.common import logger

logger = logger.setup_log(name=__name__)


# =============================================================================
# Utility
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Base Encoder
# =============================================================================

class BaseEncoder:
    """Base class shared by structure encoders and fitness encoder."""

    def __init__(self, tokenizer, cache_dir):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.mu = 0.0
        self.sigma = 1.0

    # Z-score
    def compute_zscore(self, values):
        arr = np.array(values)
        mu = arr.mean()
        sigma = arr.std() or 1.0
        return float(mu), float(sigma)

    # JSON save/load
    def dump_mu_sigma(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mu": self.mu, "sigma": self.sigma}, f, indent=4)
        logger.info(f"Saved mu={self.mu} sigma={self.sigma} → {path}")

    def load_mu_sigma(self, path):
        with open(path, "r") as f:
            d = json.load(f)
        self.mu = d["mu"]
        self.sigma = d["sigma"]
        logger.info(f"Loaded mu={self.mu} sigma={self.sigma} ← {path}")


# =============================================================================
# Structure Feature Dataset (sequence + residue-level label)
# =============================================================================

class DataEncoder(BaseEncoder):

    ENCODERS = {
        "position": "binary",
        "degree": "regression",
        "area": "regression",
        "rsa": "regression",
        "order": "regression"
    }

    def __init__(self, seq_pkl, pkl_file, tokenizer, cache_dir, task):
        super().__init__(tokenizer, cache_dir)
        self.seq_pkl = seq_pkl
        self.pkl_file = pkl_file
        self.task = task

    # Convert raw pkl → {"id", "sequence", "labels"}
    def format_raw(self, pkl_data):
        seqs = load_existing_results(self.seq_pkl)

        # Compute global z-score for regression tasks
        if self.task != 'position':
            all_vals = [v for fdict in pkl_data.values() for v in fdict.values()]
            self.mu, self.sigma = self.compute_zscore(all_vals)

        formatted = []

        for pid, feature in pkl_data.items():
            if pid not in seqs:
                continue
            seq_info = seqs[pid]
            seq = seq_info['seq']
            seq_dict = seq_info['seq_dict']
            first_id = min(seq_dict.keys())
            L = len(seq)
            labels = [-100] * L
            for res_id, v in feature.items():
                idx = res_id - first_id
                if self.task == "position":
                    labels[idx] = 1 if v > 0 else 0
                else:
                    labels[idx] = v
            formatted.append({"id": pid, "sequence": seq, "labels": labels})
        return formatted

    # Tokenize with task-specific encoders
    def encode_item(self, x):
        tok = self.tokenizer(x["sequence"], padding=False, return_attention_mask=True)
        # Add special tokens padding
        tok["labels"] = [-100] + x["labels"] + [-100]
        return tok

    def tokenize_dataset(self, dataset):
        cache = os.path.join(self.cache_dir, self.task)

        if os.path.exists(cache):
            logger.info(f"[{self.task}] Loading cached dataset: {cache}")
            return load_from_disk(cache)

        logger.info(f"[{self.task}] Tokenizing...")
        tokenized = dataset.map(
            lambda x: self.encode_item(x),
            remove_columns=["id", "sequence", "labels"],
        )
        tokenized.save_to_disk(cache)
        return tokenized

    # Main function
    def get(self):
        raw = load_existing_results(self.pkl_file)
        formatted = self.format_raw(raw)
        dataset = Dataset.from_list(formatted)
        tokenized = self.tokenize_dataset(dataset)

        # Save z-score stats for regression
        if self.task != 'position':
            self.dump_mu_sigma(os.path.join(self.cache_dir, self.task, "mu_sigma.json"))

        return tokenized, len(dataset)


# =============================================================================
# Collator with label padding
# =============================================================================

class LabelPaddingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        max_len = max(len(f["labels"]) for f in features)
        for f in features:
            f["labels"] += [-100] * (max_len - len(f["labels"]))
        return super().__call__(features)


# =============================================================================
# Structure Loader
# =============================================================================

def run_structure_encoder(seq_pkl, feature_pkl, model_dir, tokenized_cache_path, task, test_ratio, batch_size, seed):
    set_seed(seed)

    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    collator = LabelPaddingCollator(tokenizer=tokenizer)

    encoder = DataEncoder(seq_pkl, feature_pkl, tokenizer, tokenized_cache_path, task)
    tokenized, total = encoder.get()
    split = tokenized.train_test_split(test_size=test_ratio, seed=seed)

    train = DataLoader(split["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val = DataLoader(split["test"], batch_size=batch_size, shuffle=False, collate_fn=collator)

    logger.info(f"[{task}] total={total}, train={len(split['train'])}, val={len(split['test'])}")
    return train, val, tokenizer


# =============================================================================
# Fitness Dataset (wt–mt pair)
# =============================================================================

class FitnessData(BaseEncoder):
    def __init__(self, pkl_file, tokenizer, cache_dir="fitness_cache"):
        super().__init__(tokenizer, cache_dir)
        self.pkl_file = pkl_file
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def read_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def zscore(self, items):
        vals = [x["fitness"] for x in items]
        mu, sigma = self.compute_zscore(vals)

        stats_path = os.path.join(self.cache_dir, "mu_sigma.json")
        if os.path.exists(stats_path):
            self.load_mu_sigma(stats_path)
        else:
            self.mu, self.sigma = mu, sigma
            self.dump_mu_sigma(stats_path)

        for x in items:
            x["mutation"] = x["id"].split("_")[1]
            x["raw_fitness"] = x["fitness"]
            x["fitness"] = (x["fitness"] - self.mu) / self.sigma

        return items

    @staticmethod
    def encode_pair(x, tokenizer):
        wt = tokenizer(x["wt_seq"], padding=False)
        mt = tokenizer(x["mt_seq"], padding=False)

        return {
            "wt_input_ids": wt["input_ids"],
            "wt_attention_mask": wt["attention_mask"],
            "mut_input_ids": mt["input_ids"],
            "mut_attention_mask": mt["attention_mask"],
            "mutation_pos": int(x["id"].split("_")[1][1:-1]),
            "labels": torch.tensor(x["fitness"], dtype=torch.float),
        }

    def tokenize(self, dataset):
        if os.path.exists(self.cache_dir):
            try:
                return load_from_disk(self.cache_dir)
            except:
                logger.warning("Cache corrupted → regenerating.")

        tokenized = dataset.map(
            lambda x: self.encode_pair(x, self.tokenizer),
            remove_columns=["fitness", "wt_seq", "mt_seq"],
        )
        tokenized.save_to_disk(self.cache_dir)
        return tokenized

    def get(self):
        raw = self.read_pkl(self.pkl_file)
        z = self.zscore(raw)
        dataset = Dataset.from_list(z)
        tokenized = self.tokenize(dataset)
        logger.info(f"Fitness samples = {len(dataset)}")
        return tokenized


class FitnessCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        wt = self.tokenizer.pad(
            [{"input_ids": b["wt_input_ids"], "attention_mask": b["wt_attention_mask"]} for b in batch],
            return_tensors="pt",
        )
        mt = self.tokenizer.pad(
            [{"input_ids": b["mut_input_ids"], "attention_mask": b["mut_attention_mask"]} for b in batch],
            return_tensors="pt",
        )

        return {
            "wt_input_ids": wt["input_ids"],
            "wt_attention_mask": wt["attention_mask"],
            "mut_input_ids": mt["input_ids"],
            "mut_attention_mask": mt["attention_mask"],
            "mutation_pos": torch.tensor([b["mutation_pos"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
        }


def run_fitness_data(model_dir, feature_pkl, tokenized_cache_path, test_ratio, seed, batch_size):
    set_seed(seed)
    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)

    fd = FitnessData(tokenizer=tokenizer, cache_dir=tokenized_cache_path, pkl_file=feature_pkl)
    tokenized = fd.get()

    split = tokenized.train_test_split(test_size=test_ratio, seed=seed)
    collator = FitnessCollator(tokenizer)

    train = DataLoader(split["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val = DataLoader(split["test"], batch_size=batch_size, shuffle=False, collate_fn=collator)

    logger.info(f"Fitness train={len(split['train'])}, val={len(split['test'])}")
    return train, val, tokenizer

if __name__ == '__main__':
    seq_pkl = r"/Users/douzhixin/Developer/qPacking-esm/data/feature/all/test/feature_resid_name.pkl"
    feature_pkl = r"/Users/douzhixin/Developer/qPacking-esm/data/feature/all/test/feature_position.pkl"
    model_dir = r"/Users/douzhixin/Developer/qPacking-esm/data/checkpoints/esm2_t30_150M_UR50D"
    cache_dir = r"/Users/douzhixin/Developer/qPacking-esm/data/test/tokenized_cache"
    test_ratio = 0.1
    batch_size = 16
    seed = 3407
    run_structure_encoder(seq_pkl=seq_pkl, feature_pkl=feature_pkl, model_dir=model_dir,
                          tokenized_cache_path=cache_dir, task='position', test_ratio=test_ratio, batch_size=batch_size, seed=seed)