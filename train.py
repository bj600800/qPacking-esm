"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/2/14

# Description: train qPacking
# ------------------------------------------------------------------------------
"""
import argparse
from qpacking.models import dataset
from qpacking.models.setup_train import train_hydrophobic_binary_classification, train_hydrophobic_contrastive_model
from scripts.configs import Config
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)


def hydrophobic_binary(config):
    dataset_args = {
        "fasta_file": config.path.fasta_file,
        "pkl_file": config.path.pkl_file,
        "model_dir": config.path.model_dir,
        "tokenized_cache_path": config.path.tokenized_cache_path,
        "test_ratio": config.training_args.test_ratio,
        "batch_size": config.training_args.batch_size,
        "seed": config.training_args.seed,
    }

    # load data
    try:
        train_dataloader, valid_dataloader, num_clusters = dataset.run(**dataset_args)
    except Exception as e:
        logger.error("Failed to load dataset with dataset_args!")
        logger.error(str(e))
        raise

    model_args = {
        "model_dir": config.path.model_dir,
        "checkpoints_dir": config.path.checkpoints_dir,
        "logging_dir": config.path.logging_dir,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "seed": config.training_args.seed,
        "lr": config.training_args.lr,
        "num_clusters": config.training_args.num_clusters,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "eval_steps": config.training_args.eval_steps,
        "save_steps": config.training_args.save_steps,
        "eval_strategy": config.training_args.eval_strategy,
        "save_strategy": config.training_args.save_strategy,
        "logging_strategy": config.training_args.logging_strategy,
        "logging_steps": config.training_args.logging_steps,
        "save_total_limit": config.training_args.save_total_limit,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_hydrophobic_binary_classification(**model_args)
    except TypeError as e:
        logger.error("Failed to start training — argument mismatch!")
        logger.error(str(e))
        raise

def hydrophobic_contrastive(config):
    dataset_args = {
        "fasta_file": config.path.fasta_file,
        "pkl_file": config.path.pkl_file,
        "model_dir": config.path.model_dir,
        "tokenized_cache_path": config.path.tokenized_cache_path,
        "test_ratio": config.training_args.test_ratio,
        "batch_size": config.training_args.batch_size,
        "seed": config.training_args.seed,
    }

    # load data
    try:
        train_dataloader, valid_dataloader, _ = dataset.run(**dataset_args)
    except Exception as e:
        logger.error("Failed to load dataset with dataset_args!")
        logger.error(str(e))
        raise

    model_args = {
        "model_dir": config.path.model_dir,
        "checkpoints_dir": config.path.checkpoints_dir,
        "logging_dir": config.path.logging_dir,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "seed": config.training_args.seed,
        "lr": config.training_args.lr,
        "proj_dim": config.training_args.proj_dim,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "eval_steps": config.training_args.eval_steps,
        "save_steps": config.training_args.save_steps,
        "eval_strategy": config.training_args.eval_strategy,
        "save_strategy": config.training_args.save_strategy,
        "logging_strategy": config.training_args.logging_strategy,
        "logging_steps": config.training_args.logging_steps,
        "save_total_limit": config.training_args.save_total_limit,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_hydrophobic_contrastive_model(**model_args)
    except TypeError as e:
        logger.error("Failed to start training — argument mismatch!")
        logger.error(str(e))
        raise


def degree_class():
    pass


def main():
    parser = argparse.ArgumentParser(description="Protein training script")
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['hydrophobic_binary', 'hydrophobic_contrastive'],
        help="Training task selection: hydrophobic or degree"
    )

    parser.add_argument(
        '--yaml',
        type=str,
        required=True,
        help="Hyper-params file needed for specifying the training task"
    )

    args = parser.parse_args()
    yaml_path = args.yaml
    task = args.task

    config = Config.from_yaml(yaml_path, task)
    log = Config.ConfigLogger(config, task)
    log.log()
    if task == 'hydrophobic_binary':
        hydrophobic_binary(config)

    elif task == 'hydrophobic_contrastive':
        hydrophobic_contrastive(config)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()