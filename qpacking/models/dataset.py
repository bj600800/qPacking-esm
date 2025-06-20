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


def load_seq(filepath):
    sequences = {}
    for record in SeqIO.parse(filepath, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def process_raw_data(fasta_file, pkl_file):
    """
    load raw data from a fasta file and existing results from pkl file.
    create a list of labels.


    Args:
        class_feature: input feature is hydrophobic cluster class label or not.
        fasta_file: path str
        pkl_file: path str

    Returns:
        raw_data: list of dicts, each dict contains 'id', 'sequence', and 'label'

    """
    
    raw_data = []
    sequences = load_seq(fasta_file)
    pkl_data = load_existing_results(pkl_file)
    for protein_name, feature in pkl_data.items():
        if protein_name in sequences:
            example = {
                'id': protein_name,
                'sequence': sequences[protein_name],
                'label': {str(k): v for k, v in feature.items()}
            }
            raw_data.append(example)
    return raw_data


def get_clusters_num(raw_data):
    """
    Get the maximum number of clusters from the raw data.
    Args:
        raw_data: list of dicts, each dict contains 'id', 'sequence', and 'label'

    Returns:
        num_clusters: int, number of clusters
    """
    if not raw_data:
        return 0

    max_label = max({v for example in raw_data for k, v in example['label'].items() if v is not None})

    return max_label + 1  # +1 because labels are 0-indexed, so the number of clusters is max_label + 1

def encode_data(data, tokenizer):
    """
    use EsmTokenizer to encode the input data using batch encoding.
    Args:
        data: raw_data
        tokenizer: str -> EsmTokenizer

    Returns:

    """
    # Tokenized
    tokenized_input = tokenizer(data['sequence'],
                                return_attention_mask=True,
                                padding=False,
                                return_tensors=None)

    label_dict = {int(k): v for k, v in data['label'].items()}

    hydrophobic_idx = [k for k, v in label_dict.items() if v is not None]

    # init labels and attention_mask
    input_len = len(tokenized_input['input_ids'])
    labels = [-100] * input_len
    # attention_mask = [0] * input_len

    # update labels and attention_mask with label_dict and hydrophobic_idx
    for i in range(1, input_len - 1):
        seq_idx = i - 1 # sequence index starts from 0
        if seq_idx in hydrophobic_idx:
            labels[i] = 1
            # attention_mask[i] = 1  # only unmask the hydrophobic positions in case of other positions affect the experiment explanation
        else:
            labels[i] = 0
    # set <cls>, <eos> attention_mask to 1
    # attention_mask[0] = 1  # <cls>
    # attention_mask[-1] = 1  # <eos>

    tokenized_input['labels'] = labels

    # tokenized_input['attention_mask'] = attention_mask

    return tokenized_input


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


def run(fasta_file, pkl_file, model_dir, tokenized_cache_path, test_ratio, batch_size, seed):
    set_seed(seed)  # for reproducibility
    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)

    raw_data = process_raw_data(fasta_file, pkl_file)
    max_cluster_num = get_clusters_num(raw_data)

    dataset = Dataset.from_list(raw_data)

    if os.path.exists(tokenized_cache_path):
        logger.info("Tokenization history found, loading to memory...")
        tokenized_dataset = load_from_disk(tokenized_cache_path)
    else:
        tokenized_dataset = dataset.map(
            lambda x: encode_data(x, tokenizer),
            batched=False,
            remove_columns=['id', 'sequence', 'label'],  # processed data will not contain these columns from the original raw data.
            desc="Tokenizing dataset"
        )
        logger.info("Tokenization completed, saving to disk...")
        tokenized_dataset.save_to_disk(tokenized_cache_path)

    data_collator = DataCollatorWithPaddingForLabels(tokenizer=tokenizer)
    split_dataset = tokenized_dataset.train_test_split(test_size=test_ratio, seed=seed)
    train_dataloader, valid_dataloader = get_data_loader(split_dataset, data_collator, batch_size)

    return train_dataloader, valid_dataloader, max_cluster_num



if __name__ == '__main__':
    fasta_file = r"/Users/douzhixin/Developer/qPacking/data/test/sequence.fasta"
    pkl_file = r"/Users/douzhixin/Developer/qPacking/data/test/class_results.pkl"
    model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t30_150M_UR50D"
    tokenized_cache_path = None

    # configs params
    seed = 3407
    test_ratio = 0.1
    batch_size = 8

    run(fasta_file, pkl_file, model_dir, tokenized_cache_path, test_ratio, batch_size, seed)

