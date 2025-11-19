"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/5

# Description: model wrappers and training setup
# ------------------------------------------------------------------------------
"""

import os
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from qpacking_esm.model.models import TokenClassificationModel, TokenRegressionModel
from qpacking_esm.model.FitnessRegression import FitnessRegressionModel
from qpacking_esm.model import params
from qpacking_esm.model.metrics import compute_binary_metrics, compute_regression_metrics
from qpacking_esm.model.save import SaveCompleteModelCallback
from qpacking_esm.common import logger

logger = logger.setup_log(name=__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_hydrophobic_binary_classification(
        model_dir, checkpoints_dir, logging_dir,
        batch_size, num_epochs, seed, lr, num_class,
        train_dataloader, valid_dataloader,
        lora_rank, lora_alpha, lora_dropout,
        eval_steps, eval_strategy, logging_strategy,
        logging_steps, save_steps, save_strategy, save_total_limit,
        reporter, metric_for_best_model, greater_is_better, tokenizer, task):

    model = TokenClassificationModel(
        model_dir=model_dir,
        num_class=num_class,  # binary class: 1/0
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        eval_strategy=eval_strategy,
        logging_strategy=logging_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        seed=seed,
        report_to=reporter,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=valid_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        compute_metrics=compute_binary_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5),
                   SaveCompleteModelCallback(model=model, tokenizer=tokenizer)],
    )
    trainer.train()

    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    model.backbone.save_pretrained(best_model_path)

    torch.save(model.head.classifier.state_dict(), f"{best_model_path}/classifier_head.pt")

    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best fine-tuned adapter and classifier [{task}] saved to: {best_model_path}")


def train_token_regression(
        model_dir, checkpoints_dir, logging_dir,
        batch_size, num_epochs, seed, lr,
        train_dataloader, valid_dataloader,
        lora_rank, lora_alpha, lora_dropout,
        eval_steps, save_steps, eval_strategy, save_strategy, save_total_limit,
        logging_strategy, logging_steps, reporter,
        metric_for_best_model, greater_is_better, tokenizer, task):

    model = TokenRegressionModel(
        model_dir=model_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        logging_strategy=logging_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        seed=seed,
        report_to=reporter,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=valid_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        compute_metrics=compute_regression_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5),
                   SaveCompleteModelCallback(model=model, tokenizer=tokenizer)]
    )
    trainer.train()

    # Save the best model after model
    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    model.backbone.save_pretrained(best_model_path)

    torch.save(model.head.regressor.state_dict(), f"{best_model_path}/regression_head.pt")

    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best fine-tuned adapter and regressor [{task}] saved to: {best_model_path}")


def train_fitness_regression_head(model_dir, model_src, unfreeze_last_n, emb_src,
                                  checkpoints_dir, lr, eval_strategy, save_strategy, logging_strategy,
                                  save_total_limit, eval_steps, save_steps, batch_size, num_epochs,
                                  logging_dir, logging_steps, seed, reporter, metric_for_best_model, greater_is_better,
                                  train_dataloader, valid_dataloader, tokenizer):

    model = FitnessRegressionModel(model_dir, model_src, unfreeze_last_n, emb_src, params)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        logging_strategy=logging_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        seed=seed,
        report_to=reporter,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=valid_dataloader.dataset,
            compute_metrics=compute_regression_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5),
                       SaveCompleteModelCallback(model=model, tokenizer=tokenizer)]
        )
        trainer.train()
    except TypeError as e:
        pass

    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    model.model.save_pretrained(best_model_path)

    torch.save(model.regressor.state_dict(), f"{best_model_path}/regression_head.pt")

    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best trained regressor head [Fitness] saved to: {best_model_path}")

