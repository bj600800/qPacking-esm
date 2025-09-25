"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/5

# Description: Construct dataset for fine-tuning esm-2
# ------------------------------------------------------------------------------
"""
import pickle
import json
import os
import random
import numpy as np
from Bio import SeqIO

import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_from_disk

from qpacking.utils.analyze_feature import load_existing_results
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # PyTorch CUDNN reproducibility settings
    torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # disables auto-tuning for performance, ensuring reproducibility

class DataEncoder:
    def __init__(self, fasta_file, pkl_file, tokenizer, tokenized_cache_path, task):
        self.fasta_path = fasta_file
        self.pkl_file = pkl_file
        self.tokenized_cache_path = tokenized_cache_path
        self.task = task
        self.tokenizer = tokenizer
        self.mu = 0
        self.sigma = 1

    def load_seq(self):
        sequences = {}
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            sequences[record.id] = str(record.seq)
        return sequences


    def dump_mean_std_json(self):
        """
        Save the mean and std used for normalization to a JSON file.
        Args:
            feature_name: z-score feature name
        """
        filepath = os.path.join(self.tokenized_cache_path, self.task, f"{self.task}_mu_sigma.json")

        if not hasattr(self, 'mu') or not hasattr(self, 'sigma'):
            raise AttributeError("mu and sigma must be defined before dumping.")

        stats = {
            "mu": float(self.mu),
            "sigma": float(self.sigma),
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=4)

        logger.info(f"[{self.task}]: Mu and Sigma saved to {filepath}")

    def format_raw_data(self, pkl_data):
        sequences = self.load_seq()
        format_data = []

        label_values = [
            v
            for feature in pkl_data.values()
            for v in feature.values()
        ]
        label_array = np.array(label_values)
        self.mu = float(np.mean(label_array)) if label_values else 0.0
        self.sigma = float(np.std(label_array)) if label_values else 1.0
        if self.sigma == 0:
            self.sigma = 1.0

        for protein_name, feature in pkl_data.items():
            if protein_name in sequences:
                seq = sequences[protein_name]
                labels = [-100] * len(seq)
                for idx, raw_value in feature.items():
                    if self.task in ["hydrophobic_binary", "hydrophobic_contrastive"]:
                        labels[idx] = raw_value
                    else:
                        zscore_value = (raw_value - self.mu) / self.sigma
                        labels[idx] = zscore_value
                data = {
                    'id': protein_name,
                    'sequence': seq,
                    'labels': labels
                }
                format_data.append(data)

        return format_data

    @staticmethod
    def encode_hydrophobic_binary(x, tokenizer):
        """
        encode x with binary 1/0
        Args:
            x: {'id': str, 'sequence': str, 'label': dict}
            tokenizer: EsmTokenizer

        Returns:
            tokenized_input: dict with keys 'input_ids', 'attention_mask', 'labels'

        """
        # Tokenized
        tokenized_input = tokenizer(x['sequence'],
                                    return_attention_mask=True,
                                    padding=False,
                                    return_tensors=None)
        labels = x['labels']
        binary_labels = [1 if l > 0 else 0 for l in labels]
        binary_labels = [-100] + binary_labels + [-100]  # add -100 for eos and cls tokens
        tokenized_input['labels'] = binary_labels
        return tokenized_input

    @staticmethod
    def encode_hydrophobic_contrastive(x, tokenizer):
        """
        Encode x with labels: 0 for non-hydrophobic, 1-n for hydrophobic cluster numbers.
        Args:
            x: {'id': str, 'sequence': str, 'label': dict}
            tokenizer: EsmTokenizer

        Returns:
            tokenized_input: dict with keys 'input_ids', 'attention_mask', 'labels'
        """
        # Tokenized
        tokenized_input = tokenizer(x['sequence'],
                                    return_attention_mask=True,
                                    padding=False,
                                    return_tensors=None)
        labels = [i if i > 0 else -100 for i in x['labels']]  # ignore non-hydrophobic cluster residue
        labels = [-100] + labels + [-100]  # add -100 for eos and cls tokens
        tokenized_input['labels'] = labels

        return tokenized_input

    @staticmethod
    def encode_degree(x, tokenizer):
        """
        encode x with residue degree
        Args:
            x: {'id': str, 'sequence': str, 'label': dict}
            tokenizer: EsmTokenizer

        Returns:
            tokenized_input: dict with keys 'input_ids', 'attention_mask', 'labels'

        """
        # Tokenized
        tokenized_input = tokenizer(x['sequence'],
                                    return_attention_mask=True,
                                    padding=False,
                                    return_tensors=None)

        labels = [-100] + x['labels'] + [-100]  # add -100 for eos and cls tokens
        tokenized_input['labels'] = labels

        return tokenized_input

    @staticmethod
    def encode_area(x, tokenizer):
        """
        encode x with residue sasa
        Args:
            x: {'id': str, 'sequence': str, 'label': dict}
            tokenizer: EsmTokenizer

        Returns:
        """
        # Tokenize sequence
        tokenized_input = tokenizer(x['sequence'],
                                    return_attention_mask=True,
                                    padding=False,
                                    return_tensors=None)

        labels = [-100] + x['labels'] + [-100]
        tokenized_input['labels'] = labels
        return tokenized_input

    @staticmethod
    def encode_rsa(x, tokenizer):
        """
        encode x with residue sasa
        Args:
            x: {'id': str, 'sequence': str, 'label': dict}
            tokenizer: EsmTokenizer

        Returns:

        """
        # Tokenized
        tokenized_input = tokenizer(x['sequence'],
                                    return_attention_mask=True,
                                    padding=False,
                                    return_tensors=None)

        labels = [-100] + x['labels'] + [-100]  # add -100 for eos and cls tokens
        tokenized_input['labels'] = labels
        return tokenized_input

    @staticmethod
    def encode_order(x, tokenizer):
        """
        Encode x with order of residues
        Args:
            x: {'id': str, 'sequence': str, 'label': list of float}
            tokenizer: EsmTokenizer

        Returns:
            tokenized_input: dict with keys 'input_ids', 'attention_mask', 'labels'
        """
        # Tokenize sequence
        tokenized_input = tokenizer(
            x['sequence'],
            return_attention_mask=True,
            padding=False,
            return_tensors=None
        )

        labels = [-100] + x['labels'] + [-100]  # add -100 for eos and cls tokens
        tokenized_input['labels'] = labels

        return tokenized_input

    @staticmethod
    def encode_centraligy(x, tokenizer):
        """
        Encode x with centrality of residues
        Args:
            x: {'id': str, 'sequence': str, 'label': list of float}
            tokenizer: EsmTokenizer

        Returns:
            tokenized_input: dict with keys 'input_ids', 'attention_mask', 'labels'
        """
        # Tokenize sequence
        tokenized_input = tokenizer(
            x['sequence'],
            return_attention_mask=True,
            padding=False,
            return_tensors=None
        )

        labels = [-100] + x['labels'] + [-100]  # add -100 for eos and cls tokens
        tokenized_input['labels'] = labels
        return tokenized_input

    def tokenize(self, dataset):
        """
        Processes the raw data according to the specified task name and calls the appropriate encoding function.
        Args:
            dataset: huggingface-transformer dataset

        Returns: tokenized_dataset

        """
        # select task-specific encoding function
        supported_tasks = {"hydrophobic_binary", "hydrophobic_contrastive", "degree",
                           "area", "rsa", "order", "centrality"}
        if self.task not in supported_tasks:
            raise ValueError(f"Unsupported task: {self.task}. Supported tasks are: {supported_tasks}")

        if self.task == 'hydrophobic_binary':
            encode_fn = self.encode_hydrophobic_binary
        elif self.task == 'hydrophobic_contrastive':
            encode_fn = self.encode_hydrophobic_contrastive
        elif self.task == 'degree':
            encode_fn = self.encode_degree
        elif self.task == 'area':
            encode_fn = self.encode_area
        elif self.task == 'rsa':
            encode_fn = self.encode_rsa
        elif self.task == 'order':
            encode_fn = self.encode_order
        elif self.task == 'centrality':
            encode_fn = self.encode_centraligy
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        task_data_cache = os.path.join(self.tokenized_cache_path, self.task)

        if os.path.exists(task_data_cache):
            logger.info(f"Tokenization history found in {task_data_cache}, loading to memory...")
            tokenized_dataset = load_from_disk(task_data_cache)
        else:
            tokenized_dataset = dataset.map(
                lambda x, encode_fn=encode_fn: encode_fn(x, self.tokenizer),  # fix encode_fn to the current function
                batched=False,
                remove_columns=['id', 'sequence', 'labels'],  # replaced with input_ids, attention_mask, labels
                desc="Tokenizing dataset"
            )
            logger.info(f"Tokenization completed, saving to {task_data_cache}")
            tokenized_dataset.save_to_disk(task_data_cache)

        return tokenized_dataset

    def get_tokenized_dataset(self):
        pkl_data = load_existing_results(self.pkl_file)
        format_data = self.format_raw_data(pkl_data)
        dataset = Dataset.from_list(format_data)
        tokenized_dataset = self.tokenize(dataset)

        if self.task not in ["hydrophobic_binary", "hydrophobic_contrastive"]:
            self.dump_mean_std_json()

        return tokenized_dataset, len(dataset)


class DataCollatorWithPaddingForLabels(DataCollatorWithPadding):
    def __call__(self, features):
        label_name = "labels"
        if label_name in features[0]:  # only need checking the first one as for unified batch shape
            max_len = max(len(f[label_name]) for f in features)
            for f in features:
                padding_len = max_len - len(f[label_name])
                f[label_name] += [-100] * padding_len
        return super().__call__(features)


def get_data_loader(split_dataset, data_collator, batch_size):
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

    train_set = dataset_dict['train']
    validation_set = dataset_dict['validation']

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    valid_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    return train_dataloader, valid_dataloader


def run_structure_encoder(fasta_file, pkl_file, model_dir, tokenized_cache_path, task, test_ratio, batch_size, seed):
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"PKL file not found: {pkl_file}")

    set_seed(seed)  # for reproducibility

    try:
        tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {model_dir}: {e}")

    data_collator = DataCollatorWithPaddingForLabels(tokenizer=tokenizer)
    data_processor = DataEncoder(fasta_file, pkl_file, tokenizer, tokenized_cache_path, task)
    tokenized_dataset, len_dataset= data_processor.get_tokenized_dataset()
    split_dataset = tokenized_dataset.train_test_split(test_size=test_ratio, seed=seed)
    train_dataloader, valid_dataloader = get_data_loader(split_dataset, data_collator, batch_size)

    logger.info(f"Total samples in dataset: {len_dataset}")
    logger.info(f"Training set size: {len(split_dataset['train'])}, Validation set size: {len(split_dataset['test'])}")
    logger.info(
        f"Batch size: {batch_size}, "
        f"Number of training batches: {len(train_dataloader)}, "
        f"Number of validation batches: {len(valid_dataloader)}")

    return train_dataloader, valid_dataloader, tokenizer


class FitnessData:
    def __init__(self, pkl_file, tokenized_cache_path, tokenizer):
        self.pkl_file = pkl_file
        self.tokenized_cache_path = tokenized_cache_path
        self.tokenizer = tokenizer
        self.mu = 0.0
        self.sigma = 1.0

    @staticmethod
    def read_pkl(pkl_file):
        with open(pkl_file, 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data

    def dump_mean_std_json(self, filepath):
        """
        Save the mean and std used for normalization to a JSON file.
        """
        if not hasattr(self, 'mu') or not hasattr(self, 'sigma'):
            raise AttributeError("mu and sigma must be defined before dumping.")

        stats = {
            "mu": float(self.mu),
            "sigma": float(self.sigma),
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=4)

        logger.info(f"Mu and Sigma saved to {filepath}")

    def load_mean_std_json(self, filepath):
        """
        Load mean and std from existing JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found. Cannot load mean/std.")

        with open(filepath, 'r') as f:
            stats = json.load(f)

        self.mu = stats.get("mu", 0.0)
        self.sigma = stats.get("sigma", 1.0)
        logger.info(f"Loaded mu={self.mu}, sigma={self.sigma} from {filepath}")

    def zscore_raw_data(self, pkl_data):
        """
        Normalize the fitness values in-place using z-score.
        """
        label_values = [item['fitness'] for item in pkl_data if 'fitness' in item]
        if not label_values:
            raise ValueError("No 'fitness' values found in data.")

        mu_sigma_path = os.path.join(self.tokenized_cache_path, "mu_sigma.json")

        if not os.path.exists(mu_sigma_path):
            label_array = np.array(label_values)
            self.mu = float(np.mean(label_array))
            self.sigma = float(np.std(label_array))
            if self.sigma == 0:
                self.sigma = 1.0  # avoid division by zero
            self.dump_mean_std_json(mu_sigma_path)
        else:
            self.load_mean_std_json(mu_sigma_path)

        for item in pkl_data:
            if 'fitness' in item:
                item['mutation'] = item['id'].split('_')[1]  # extract mutation from id
                item['raw_fitness'] = item['fitness']  # keep original fitness value
                item['fitness'] = (item['fitness'] - self.mu) / self.sigma


        logger.info(f"Z-score normalization applied: mu={self.mu:.4f}, sigma={self.sigma:.4f}")
        return pkl_data

    @staticmethod
    def encode_fn(x, tokenizer):
        wt_encode = tokenizer(
            x["wt_seq"],
            padding=False,
            truncation=False,
            return_tensors=None
        )
        mt_encode = tokenizer(
            x["mt_seq"],
            padding=False,
            truncation=False,
            return_tensors=None
        )

        assert isinstance(wt_encode["input_ids"], list), "wt input_ids is not list"
        assert isinstance(mt_encode["input_ids"], list), "mt input_ids is not list"
        assert wt_encode["input_ids"] is not None
        assert mt_encode["input_ids"] is not None


        return {
            "wt_input_ids": wt_encode["input_ids"],
            "wt_attention_mask": wt_encode["attention_mask"],
            "mut_input_ids": mt_encode["input_ids"],
            "mut_attention_mask": mt_encode["attention_mask"],
            "mutation_pos": int(x['id'].split('_')[1][1:-1]),
            "label": torch.tensor(x["fitness"], dtype=torch.float),
        }

    @staticmethod
    def is_valid_tokenized_cache(path):
        try:
            dataset = load_from_disk(path)
            return isinstance(dataset, Dataset) or 'train' in dataset
        except Exception as e:
            logger.warning(f"Cache at {path} is invalid or corrupted: {e}")
            return False

    def tokenize(self, dataset):
        try:
            tokenized_dataset = load_from_disk(self.tokenized_cache_path)
        except:
            logger.info(f"No valid tokenized dataset found. Starting fresh tokenization...")
            tokenized_dataset = dataset.map(
                lambda x: self.encode_fn(x, self.tokenizer),
                remove_columns=['fitness', 'wt_seq', 'mt_seq'],  # remove original columns
                desc="Tokenizing dataset",
                batched=False
            )

            logger.info(f"Tokenization completed, saving to {self.tokenized_cache_path}")
            tokenized_dataset.save_to_disk(self.tokenized_cache_path)
        return tokenized_dataset

    def get_tokenized_dataset(self):
        """
        Returns a HuggingFace Dataset with normalized fitness values.
        """
        pkl_data = self.read_pkl(self.pkl_file)
        zscore_data = self.zscore_raw_data(pkl_data)
        dataset = Dataset.from_list(zscore_data)
        tokenized_dataset = self.tokenize(dataset)
        logger.info(f"Total samples in dataset: {len(dataset)}.")
        return tokenized_dataset

class FitnessCollator:
    """
    Custom collator for fitness data, handling separate tokenization for wild-type (wt) and mutant (mt) sequences.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        wt_batch = [{"input_ids": item["wt_input_ids"], "attention_mask": item["wt_attention_mask"]} for item in batch]
        mt_batch = [{"input_ids": item["mut_input_ids"], "attention_mask": item["mut_attention_mask"]} for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)

        # padding for wt and mt separately
        wt_padded = self.tokenizer.pad(wt_batch, return_tensors="pt")
        mt_padded = self.tokenizer.pad(mt_batch, return_tensors="pt")

        mutation_pos = [int(item["mutation"][1:-1]) for item in batch]  # +1 for cls token, -1 for res renumber
        mutation_pos = torch.tensor(mutation_pos, dtype=torch.long)

        return {
            "wt_input_ids": wt_padded["input_ids"],
            "wt_attention_mask": wt_padded["attention_mask"],
            "mut_input_ids": mt_padded["input_ids"],
            "mut_attention_mask": mt_padded["attention_mask"],
            "mutation_pos": torch.tensor(mutation_pos),
            "labels": labels,
        }

def run_fitness_data(model_dir, pkl_file, tokenized_cache_path, test_ratio, seed, batch_size):
    set_seed(seed)
    try:
        tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {model_dir}: {e}")
    fitness = FitnessData(pkl_file, tokenized_cache_path, tokenizer)
    tokenized_dataset = fitness.get_tokenized_dataset()
    split_dataset = tokenized_dataset.train_test_split(test_size=test_ratio, seed=seed)
    collator = FitnessCollator(tokenizer=tokenizer)
    train_dataloader, valid_dataloader = get_data_loader(split_dataset, collator, batch_size)
    torch.set_printoptions(threshold=torch.inf)
    logger.info(f"Training set size: {len(split_dataset['train'])}, Validation set size: {len(split_dataset['test'])}")
    logger.info(
        f"Batch size: {batch_size}, "
        f"Number of training batches: {len(train_dataloader)}, "
        f"Number of validation batches: {len(valid_dataloader)}")

    # batch = next(iter(train_dataloader))
    # print(batch)
    # print(batch["mutation_pos"][8])
    # print(batch["wt_input_ids"][8])
    # print(batch["mt_input_ids"][8])
    return train_dataloader, valid_dataloader, tokenizer


# def decode_sequences(batch, tokenizer):
#     input_ids = batch["input_ids"]
#     for i, ids in enumerate(input_ids):
#         # 将 Tensor 转为 list
#         if isinstance(ids, torch.Tensor):
#             ids = ids.tolist()
#         decoded = tokenizer.decode(ids, skip_special_tokens=True)
#         print(f"Sample {i} sequence:\n{decoded}\n")

# def inverse_transform(zscore_vals, mean, std, epsilon=1e-6):
#     zscore_vals = np.array(zscore_vals)
#     log_vals = zscore_vals * std + mean
#     order_vals = np.exp(log_vals) - epsilon
#     return order_vals

if __name__ == '__main__':
    # fasta_file = r"/Users/douzhixin/Developer/qPacking/data/sequence/complete_tim_80.fasta"
    # pkl_file = r"/Users/douzhixin/Developer/qPacking/data/feature/80/80_results_degree.pkl"
    # model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t30_150M_UR50D"
    # tokenized_cache_path = r"/Users/douzhixin/Developer/qPacking/data/feature/80/tokenized_cache"
    # task = r"degree"
    # test_ratio = 0.1
    # batch_size = 16
    # seed = 3407
    # run_data_encoder(fasta_file, pkl_file, model_dir, tokenized_cache_path, task, test_ratio, batch_size, seed)
    model_dir = r"/checkpoints/esm2_t30_150M_UR50D"
    pkl_file = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/tm.pkl"
    tokenized_cache_path = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/tm_tokenized_cache"
    run_fitness_data(model_dir, pkl_file, tokenized_cache_path, test_ratio=0.2, seed=3407, batch_size=16)