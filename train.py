"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/2/14

# Description: train qPacking
# ------------------------------------------------------------------------------
"""
import pprint
import argparse
from qpacking.models import dataset
from qpacking.models.setup_train import train_cluster_classification
from scripts.configs.config import Config
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)


class ConfigLogger:
    def __init__(self, config, task: str, logger=logger):
        self.config = config
        self.task = task.lower()
        self.logger = logger

    def log(self):
        self.logger.info(f"\n{'='*10} [Task: {self.task}] Config Summary {'='*10}")
        self._log_common()

        if self.task == "hydrophobic":
            self._log_hydrophobic()
        elif self.task == "mutation_effect":
            self._log_mutation_effect()
        elif self.task == "stability_prediction":
            self._log_stability_prediction()
        else:
            self.logger.warning(f"Unknown task: {self.task}. Logging only common parameters.")
        self.logger.info(f"\n{'=' * 10} End of Config Summary {'=' * 10}")

    def _log_common(self):
        cfg = self.config
        self.logger.info(f"{'='*10}[Path]{'='*10}")
        for k, v in vars(cfg.path).items():
            self.logger.info(f"{k}: {v}")

        self.logger.info(f"{'='*10}[Training Args]{'='*10}")
        for k, v in vars(cfg.training_args).items():
            self.logger.info(f"{k}: {v}")

        self.logger.info(f"{'='*10}[LoRA]{'='*10}")
        for k, v in vars(cfg.lora).items():
            self.logger.info(f"{k}: {v}")

    def _log_hydrophobic(self):
        self.logger.info("[Hydrophobic Task Specific Config]")
        # 可选打印 hydrophobic 专用参数，例如：
        # self.logger.info(f"hydrophobic_threshold: {self.config.task_args.hydrophobic_threshold}")
        pass

    def _log_mutation_effect(self):
        self.logger.info("[Mutation Effect Task Specific Config]")
        # 例如：
        # self.logger.info(f"mutation_sites: {self.config.task_args.mutation_sites}")
        pass

    def _log_stability_prediction(self):
        self.logger.info("[Stability Prediction Task Specific Config]")
        # 例如：
        # self.logger.info(f"use_ensemble: {self.config.task_args.use_ensemble}")
        pass


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
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "eval_steps": config.training_args.eval_steps,
        "save_steps": config.training_args.save_steps,
        "eval_strategy": config.training_args.eval_strategy,
        "save_strategy": config.training_args.save_strategy,
        "logging_steps": config.training_args.logging_steps,
        "save_total_limit": config.training_args.save_total_limit,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_cluster_classification(**model_args)
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
        choices=['hydrophobic', 'degree'],
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

    config = Config.from_yaml(yaml_path)
    log = ConfigLogger(config, task)
    log.log()
    if task == 'hydrophobic':
        hydrophobic_binary(config)
    elif task == 'degree':
        degree_class()
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()