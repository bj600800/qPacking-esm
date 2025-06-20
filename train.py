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
from qpacking.models.setup_train import train_cluster_classification
from scripts.configs.config import Config

def hydrophobic_binary(config):
    # path args
    model_dir = config.path.model_dir
    checkpoints_dir = config.path.checkpoints_dir
    logging_dir = config.path.logging_dir
    tokenized_cache_path = config.path.tokenized_cache_path
    fasta_file = config.path.fasta_file
    pkl_file = config.path.pkl_file

    # hyperparams
    test_ratio = config.training_args.test_ratio
    batch_size = config.training_args.batch_size
    num_epochs = config.training_args.num_epochs
    seed = config.training_args.seed
    lr = config.training_args.lr
    lora_rank = config.lora.rank
    lora_alpha = config.lora.alpha
    lora_dropout = config.lora.dropout
    eval_steps = config.training_args.eval_steps
    save_steps = config.training_args.save_steps
    eval_strategy = config.training_args.eval_strategy
    save_strategy = config.training_args.save_strategy
    logging_steps = config.training_args.logging_steps
    save_total_limit = config.training_args.save_total_limit
    reporter = config.training_args.reporter
    metric_for_best_model = config.training_args.metric_for_best_model
    greater_is_better = config.training_args.greater_is_better

    # load dataset
    train_dataloader, valid_dataloader, num_clusters = dataset.run(
        fasta_file, pkl_file, model_dir, tokenized_cache_path, test_ratio, batch_size, seed
    )

    # train
    train_cluster_classification(
        model_dir=model_dir,
        checkpoints_dir=checkpoints_dir,
        logging_dir=logging_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        seed=seed,
        lr=lr,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        eval_steps=eval_steps,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        reporter=reporter,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better
    )

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

    args = parser.parse_args()


    if args.task == 'hydrophobic':
        config = Config.from_yaml("./scripts/configs/hydrophobic_binary.yaml")
        hydrophobic_binary(config)

    elif args.task == 'degree':
        degree_class()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()