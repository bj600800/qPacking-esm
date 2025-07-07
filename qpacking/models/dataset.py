"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/5

# Description: Construct dataset for fine-tuning esm-2
# ------------------------------------------------------------------------------
"""
import os
import random
import numpy as np
from Bio import SeqIO

import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_from_disk

from qpacking.utils.analyze_pkl import load_existing_results
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

    def load_seq(self):
        sequences = {}
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            sequences[record.id] = str(record.seq)
        return sequences


    def format_raw_data(self, pkl_data):
        sequences = self.load_seq()
        format_data = []
        for protein_name, feature in pkl_data.items():
            if protein_name in sequences:
                seq = sequences[protein_name]
                labels = [0] * len(seq)  # Initialize all labels to 0
                for k, v in feature.items():
                    labels[k] = v
                data = {
                    'id': protein_name,
                    'sequence': sequences[protein_name],
                    'labels': labels
                }
                format_data.append(data)
        return format_data

    @staticmethod
    def encode_binary(x, tokenizer):
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
    def encode_contrastive(x, tokenizer):
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
        labels = x['labels']
        labels = [-100] + labels + [-100]  # add -100 for eos and cls tokens
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
        supported_tasks = {"hydrophobic_binary", "hydrophobic_contrastive", "some_other_task"}
        if self.task not in supported_tasks:
            raise ValueError(f"Unsupported task: {self.task}. Supported tasks are: {supported_tasks}")

        if self.task == 'hydrophobic_binary':
            encode_fn = self.encode_binary
        elif self.task == 'hydrophobic_contrastive':
            encode_fn = self.encode_contrastive
        elif self.task == 'some_other_task':
            raise NotImplementedError("some_other_task is not yet implemented.")
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


def run(fasta_file, pkl_file, model_dir, tokenized_cache_path, task, test_ratio, batch_size, seed):
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
    tokenized_dataset, len_dataset = data_processor.get_tokenized_dataset()
    split_dataset = tokenized_dataset.train_test_split(test_size=test_ratio, seed=seed)
    train_dataloader, valid_dataloader = get_data_loader(split_dataset, data_collator, batch_size)

    logger.info(f"Total samples in dataset: {len_dataset}")
    logger.info(f"Training set size: {len(split_dataset['train'])}, Validation set size: {len(split_dataset['test'])}")
    logger.info(
        f"Batch size: {batch_size}, "
        f"Number of training batches: {len(train_dataloader)}, "
        f"Number of validation batches: {len(valid_dataloader)}")

    return train_dataloader, valid_dataloader