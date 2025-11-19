"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/2/14

# Description: train qPacking-esm
# ------------------------------------------------------------------------------
"""
import os
import argparse
import mlflow
from datetime import datetime

from qpacking_esm.data import dataset
from qpacking_esm.model.setup_train import (train_hydrophobic_binary_classification,
                                        train_token_regression, train_fitness_regression_head)

from train_configs import Config
from qpacking_esm.common import logger

logger = logger.setup_log(name=__name__)


def hydrophobic_binary(config):
    task = config.training_args.task
    dataset_args = {
        "seq_pkl": config.path.seq_pkl,
        "feature_pkl": config.path.feature_pkl,
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
        "add_lora_layers": config.lora.add_lora_layers,
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
        logger.error("Failed to start model — argument mismatch!")
        logger.error(str(e))
        raise

def token_regression(config):
    task = config.training_args.task
    dataset_args = {
        "seq_pkl": config.path.seq_pkl,
        "feature_pkl": config.path.feature_pkl,
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
        "add_lora_layers": config.lora.add_lora_layers,
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
        logger.error("Failed to start model — argument mismatch!")
        logger.error(str(e))
        raise

def fitness_regression(config):
    task = config.training_args.task
    dataset_args = {
        "model_dir": config.path.model_dir,
        "feature_pkl": config.path.feature_pkl,
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
        logger.error("Failed to start model — argument mismatch!")
        logger.error(str(e))
        raise

def create_fitness_mlflow_experiment(config, task):
    """
    Create an MLflow experiment for the given task.
    """
    model_src = config.path.model_src
    pkl_name = os.path.basename(config.path.feature_pkl).split('.')[0]
    unfrozen_layers = config.training_args.unfreeze_last_n
    emb_src = config.training_args.emb_src
    if model_src == "official":
        base_model_name = os.path.basename(config.path.model_dir)
    else:
        base_model_name = os.path.basename(os.path.dirname(config.path.model_dir))

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

    timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
    run_name = f"{timestamp}_{task}_{base_model_name}_{pkl_name}_unfrozen:{unfrozen_layers}_{emb_src}"
    logger.info(f"MLflow run name: {run_name}")

    return run_name, task, model_src, base_model_name, pkl_name, unfrozen_layers, emb_src

def create_mlflow_experiment(config, task):
    """
    Create an MLflow experiment for position / degree / bsa / rsa / order.
    """
    pkl_name = os.path.basename(config.path.feature_pkl).split('.')[0]
    base_model_name = os.path.basename(config.path.model_dir)

    experiment_name = f"{task}_finetune"
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        exp = mlflow.get_experiment_by_name(experiment_name)
        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
        raise

    timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
    run_name = (
        f"{timestamp}_{task}_{base_model_name}_"
        f"{pkl_name}_"
        f"lora_layers:{config.lora.add_lora_layers}_"
        f"bs:{config.training_args.batch_size}_"
        f"epoch:{config.training_args.num_epochs}"
    )

    return {
        "run_name": run_name,
        "task": task,
        "model": base_model_name,
        "pkl_name": pkl_name,
        "lr": config.training_args.lr,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout
    }

def run_fitness_with_mlflow(config):
    """
    Run the fitness regression task with MLflow experiment creation,
    run name formatting, tags, and training wrapped into a single function.
    """
    task = config.training_args.task

    # --------------------------
    # Build experiment name
    # --------------------------
    model_src = config.path.model_src
    pkl_name = os.path.basename(config.path.feature_pkl).split('.')[0]
    unfrozen_layers = config.training_args.unfreeze_last_n
    emb_src = config.training_args.emb_src

    if model_src == "official":
        base_model_name = os.path.basename(config.path.model_dir)
    else:
        base_model_name = os.path.basename(os.path.dirname(config.path.model_dir))

    experiment_name = f"{task}_{base_model_name}_{pkl_name}"

    # --------------------------
    # Set MLflow experiment
    # --------------------------
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
        logger.error(f"Experiment existed but deleted, restored: {experiment_name}")
        raise

    # --------------------------
    # Run name
    # --------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
    run_name = f"{timestamp}_{task}_{base_model_name}_{pkl_name}_unfrozen:{unfrozen_layers}_{emb_src}"
    logger.info(f"MLflow run name: {run_name}")

    # --------------------------
    # Start MLflow Run
    # --------------------------
    with mlflow.start_run(run_name=run_name):

        mlflow.set_tags({
            "task": task,
            "model_src": model_src,
            "model": base_model_name,
            "pkl_name": pkl_name,
            "unfrozen_layers": unfrozen_layers,
            "emb_src": emb_src,
        })
        logger.info("MLflow tags set for fitness regression.")

        # ----------------------
        # Run training
        # ----------------------
        fitness_regression(config)

def run_hydrophobic_binary_with_mlflow(config):
    info = create_mlflow_experiment(config, config.training_args.task)

    with mlflow.start_run(run_name=info["run_name"]):
        mlflow.set_tags(info)

        hydrophobic_binary(config)

def run_token_regression_with_mlflow(config):
    info = create_mlflow_experiment(config, config.training_args.task)

    with mlflow.start_run(run_name=info["run_name"]):
        mlflow.set_tags(info)

        token_regression(config)

def main():
    parser = argparse.ArgumentParser(description="Protein model script")
    parser.add_argument(
        '--yaml',
        type=str,
        required=True,
        help="Hyper-params file needed for specifying the model task"
    )

    args = parser.parse_args()
    yaml_path = args.yaml

    config, task = Config.from_yaml(yaml_path)
    log = Config.ConfigLogger(config, task)
    log.log()
    if task == 'position':
        run_hydrophobic_binary_with_mlflow(config)

    elif task in ['degree', 'bsa', 'rsa', 'order']:
        run_token_regression_with_mlflow(config)

    elif task == 'fitness':
        run_fitness_with_mlflow(config)

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()