"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/2/14

# Description: train qPacking
# ------------------------------------------------------------------------------
"""

import os
import argparse
import mlflow
from datetime import datetime

from qpacking.models import dataset
from qpacking.models.setup_train import (train_hydrophobic_binary_classification, train_hydrophobic_contrastive_model,
                                         train_token_regression, train_fitness_regression_head)
from scripts.configs import Config
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)


def hydrophobic_binary(config, task):
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
        train_dataloader, valid_dataloader, tokenizer = dataset.run_structure_encoder(**dataset_args, task=task)
    except Exception as e:
        logger.error("Failed to load dataset with dataset_args!")
        logger.error(str(e))
        raise

    model_args = {
        "model_dir": config.path.model_dir,
        "checkpoints_dir": os.path.join(config.path.checkpoints_dir, task),
        "logging_dir": config.path.logging_dir,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "seed": config.training_args.seed,
        "lr": config.training_args.lr,
        "num_class": config.training_args.num_class,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "eval_steps": config.training_args.eval_steps,
        "eval_strategy": config.training_args.eval_strategy,
        "save_total_limit": config.training_args.save_total_limit,
        "save_steps": config.training_args.save_steps,
        "save_strategy": config.training_args.save_strategy,
        "logging_strategy": config.training_args.logging_strategy,
        "logging_steps": config.training_args.logging_steps,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_hydrophobic_binary_classification(**model_args, tokenizer=tokenizer, task=task)
    except TypeError as e:
        logger.error("Failed to start training — argument mismatch!")
        logger.error(str(e))
        raise

def hydrophobic_contrastive(config, task):
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
        train_dataloader, valid_dataloader, tokenizer = dataset.run_structure_encoder(**dataset_args, task=task)
    except Exception as e:
        logger.error("Failed to load dataset with dataset_args!")
        logger.error(str(e))
        raise

    model_args = {
        "model_dir": config.path.model_dir,
        "checkpoints_dir": os.path.join(config.path.checkpoints_dir, task),
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
        "save_total_limit": config.training_args.save_total_limit,
        "eval_strategy": config.training_args.eval_strategy,
        "save_strategy": config.training_args.save_strategy,
        "logging_strategy": config.training_args.logging_strategy,
        "logging_steps": config.training_args.logging_steps,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_hydrophobic_contrastive_model(**model_args, tokenizer=tokenizer, task=task)
    except TypeError as e:
        logger.error("Failed to start training — argument mismatch!")
        logger.error(str(e))
        raise


def token_regression(config, task):
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
        train_dataloader, valid_dataloader, tokenizer = dataset.run_structure_encoder(**dataset_args, task=task)
    except Exception as e:
        logger.error("Failed to load dataset with dataset_args!")
        logger.error(str(e))
        raise

    model_args = {
        "model_dir": config.path.model_dir,
        "checkpoints_dir": os.path.join(config.path.checkpoints_dir, task),
        "logging_dir": config.path.logging_dir,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "seed": config.training_args.seed,
        "lr": config.training_args.lr,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "eval_steps": config.training_args.eval_steps,
        "save_steps": config.training_args.save_steps,
        "save_total_limit": config.training_args.save_total_limit,
        "eval_strategy": config.training_args.eval_strategy,
        "save_strategy": config.training_args.save_strategy,
        "logging_strategy": config.training_args.logging_strategy,
        "logging_steps": config.training_args.logging_steps,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_token_regression(**model_args, tokenizer=tokenizer, task=task)
    except TypeError as e:
        logger.error("Failed to start training — argument mismatch!")
        logger.error(str(e))
        raise

def fitness_regression(config, task):
    dataset_args = {
        "model_dir": config.path.fasta_file,
        "pkl_file": config.path.pkl_file,
        "tokenized_cache_path": config.path.tokenized_cache_path,
        "test_ratio": config.training_args.test_ratio,
        "seed": config.training_args.seed,
        "batch_size": config.training_args.batch_size
    }

    # load data
    try:
        train_dataloader, valid_dataloader, tokenizer = dataset.run_fitness_data(**dataset_args)

    except Exception as e:
        logger.error("Failed to load dataset with dataset_args!")
        logger.error(str(e))
        raise

    if config.path.model_src == "official":
        base_model_name = os.path.basename(config.path.model_dir)
    else:
        base_model_name = os.path.basename(os.path.dirname(config.path.model_dir))

    model_args = {
        "model_dir": config.path.model_dir,
        "model_src": config.path.model_src,
        "unfreeze_last_n": config.training_args.unfreeze_last_n,
        "emb_src": config.training_args.emb_src,
        "checkpoints_dir": os.path.join(config.path.checkpoints_dir, task+'/'+base_model_name),
        "logging_dir": config.path.logging_dir,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "seed": config.training_args.seed,
        "lr": config.training_args.lr,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "eval_steps": config.training_args.eval_steps,
        "save_steps": config.training_args.save_steps,
        "save_total_limit": config.training_args.save_total_limit,
        "eval_strategy": config.training_args.eval_strategy,
        "save_strategy": config.training_args.save_strategy,
        "logging_strategy": config.training_args.logging_strategy,
        "logging_steps": config.training_args.logging_steps,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_fitness_regression_head(**model_args, tokenizer=tokenizer)
    except TypeError as e:
        logger.error("Failed to start training — argument mismatch!")
        logger.error(str(e))
        raise

def create_fitness_mlflow_experiment(config, task):
    """
    Create an MLflow experiment for the given task.
    """
    model_src = config.path.model_src
    pkl_name = os.path.basename(config.path.pkl_file).split('.')[0]
    unfrozen_layers = config.training_args.unfreeze_last_n
    emb_src = config.training_args.emb_src
    if model_src == "official":
        base_model_name = os.path.basename(config.path.model_dir)
    else:
        base_model_name = os.path.basename(os.path.dirname(config.path.model_dir))

    # 组装实验名
    experiment_name = f"{task}_{base_model_name}_{pkl_name}"
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except mlflow.exceptions.MlflowException as e:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
        logger.error(f"Failed to set MLflow experiment: {experiment_name}")
        logger.error(f"Rerun the script again")
        raise

    # 生成 run name（加时间戳）
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
    run_name = f"{timestamp}_{task}_{base_model_name}_{pkl_name}_unfrozen:{unfrozen_layers}_{emb_src}"
    logger.info(f"MLflow run name: {run_name}")

    return run_name, task, model_src, base_model_name, pkl_name, unfrozen_layers, emb_src


def main():
    parser = argparse.ArgumentParser(description="Protein training script")
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['hydrophobic_binary', 'hydrophobic_contrastive', 'degree', 'area',
                 'rsa', 'order', 'centrality', 'fitness'],
        help="Training task selection: [hydrophobic_binary, hydrophobic_contrastive]"
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
        hydrophobic_binary(config, task=task)

    elif task == 'hydrophobic_contrastive':
        hydrophobic_contrastive(config, task=task)

    elif task == 'degree':
        token_regression(config, task=task)

    elif task == 'area':
        token_regression(config, task=task)

    elif task == 'rsa':
        token_regression(config, task=task)

    elif task == 'order':
        token_regression(config, task=task)

    elif task == 'centrality':
        token_regression(config, task=task)

    elif task == 'fitness':
        run_name, task, model_src, base_model_name, pkl_name, unfrozen_layers, emb_src = create_fitness_mlflow_experiment(config, task)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "task": task,
                "model_src": model_src,
                "pkl_name": pkl_name,
                "unfrozen_layers": unfrozen_layers,
                "emb_src": emb_src,
                "model": base_model_name
            })
            logger.info(f"MLflow set tags: task, model_src, pkl_name, unfrozen_layers, emb_src, model")
            fitness_regression(config, task=task)

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()