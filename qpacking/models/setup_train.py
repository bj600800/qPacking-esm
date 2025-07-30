"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/5

# Description: setup trainer for fine-tuning esm-2
# ------------------------------------------------------------------------------
"""

import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)

from qpacking.models.model import (TokenClassificationModel, FocalLoss, FitnessRegressionModel,
                                   HydrophobicContrastiveModel, TokenRegressionModel)
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_gamma=2.0, focal_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)

        if focal_alpha is not None and not isinstance(focal_alpha, torch.Tensor):
            focal_alpha = torch.tensor(focal_alpha, dtype=torch.float32)

        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
            ignore_index=-100
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        logits = logits.to(device)
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_binary_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    true_predictions = []
    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            if l != -100:
                true_labels.append(l)
                true_predictions.append(p)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='macro', zero_division=0
    )

    acc = accuracy_score(true_labels, true_predictions)

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }

def compute_regression_metrics(eval_pred):
    """
    用于 Huggingface Trainer 的回归评估函数
    Args:
        eval_pred: EvalPrediction 对象，包含 predictions 和 label_ids
    Returns:
        dict: 各类回归指标
    """
    predictions, labels = eval_pred
    # 对于逐位点回归任务，预测 shape 应为 (batch_size, seq_len, 1) 或 (batch_size, seq_len)
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)

    # 将 -100 的标签去除（mask）
    mask = labels != -100
    y_true = labels[mask]
    y_pred = predictions[mask]

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # R²
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_total if ss_total > 0 else float("nan")

    # Pearson & Spearman
    pearson_corr = pearsonr(y_true, y_pred)[0]
    spearman_corr = spearmanr(y_true, y_pred)[0]

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr
    }

# def compute_regression_metrics(pred):
#     preds = pred.predictions.squeeze()  # shape: (batch,)
#     labels = pred.label_ids.squeeze()
#
#     # 反归一化
#     mu_sigma_path = os.path.join(config.path.tokenized_cache_path, "mu_sigma.json")
#     with open(mu_sigma_path, "r") as f:
#         stats = json.load(f)
#     mu = stats["mu"]
#     sigma = stats["sigma"]
#
#     preds_raw = preds * sigma + mu
#     labels_raw = labels * sigma + mu
#
#     return {
#         "pearsonr": pearsonr(preds_raw, labels_raw).statistic,
#         "spearmanr": spearmanr(preds_raw, labels_raw).statistic,
#         "mse_raw": mean_squared_error(preds_raw, labels_raw)
#     }


class SaveCompleteModelCallback(TrainerCallback):
    """
    saved 1
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = args.output_dir
        output_dir = os.path.join(checkpoint_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)

        # save adapter
        if hasattr(self.model, "model") and hasattr(self.model.model, "save_pretrained"):
            self.model.model.save_pretrained(output_dir)

        if hasattr(self.model, "classifier"):
            torch.save(self.model.classifier.state_dict(), os.path.join(output_dir, "classifier_head.pt"))

        if hasattr(self.model, "regressor"):
            torch.save(self.model.regressor.state_dict(), os.path.join(output_dir, "regression_head.pt"))

        self.tokenizer.save_pretrained(output_dir)

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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
                   SaveCompleteModelCallback(model=model, tokenizer=tokenizer)],
    )
    trainer.train()

    # Save the best model after training
    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    # 保存 PEFT (LoRA adapter)
    model.model.save_pretrained(best_model_path)

    # 保存分类头
    torch.save(model.classifier.state_dict(), f"{best_model_path}/classifier_head.pt")

    # 保存 tokenizer（建议你提前加载并传入 tokenizer）
    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best fine-tuned adapter and classifier [{task}] saved to: {best_model_path}")


def train_hydrophobic_contrastive_model(
        model_dir, lora_rank, lora_alpha, lora_dropout, proj_dim,
        checkpoints_dir, lr, eval_strategy, save_strategy, logging_strategy,
        eval_steps, save_steps, save_total_limit,
        batch_size, num_epochs, logging_dir, logging_steps,
        seed, reporter, metric_for_best_model, greater_is_better,
        train_dataloader, valid_dataloader, tokenizer, task):
    """
    setup contrastive model trainer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HydrophobicContrastiveModel(
        model_dir=model_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        proj_dim=proj_dim
    ).to(device)


    # 设置 Trainer
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
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
        max_grad_norm=1.0  # add gradient clipping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=valid_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
                   SaveCompleteModelCallback(model=model, tokenizer=tokenizer)],
    )

    trainer.train()

    # Saved 2:
    # save the best model after training
    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    # 保存 PEFT (LoRA adapter)
    model.model.save_pretrained(best_model_path)

    # 保存 tokenizer（建议你提前加载并传入 tokenizer）
    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best fine-tuned adapter [{task}] saved to: {best_model_path}")


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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
                   SaveCompleteModelCallback(model=model, tokenizer=tokenizer)]
    )
    trainer.train()

    # Save the best model after training
    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    # 保存 PEFT (LoRA adapter)
    model.model.save_pretrained(best_model_path)

    # 保存分类头
    torch.save(model.regressor.state_dict(), f"{best_model_path}/regression_head.pt")

    # 保存 tokenizer（建议你提前加载并传入 tokenizer）
    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best fine-tuned adapter and regressor [{task}] saved to: {best_model_path}")


def train_fitness_regression_head(model_dir, model_src,
                                  checkpoints_dir, lr, eval_strategy, save_strategy, logging_strategy,
                                  save_total_limit, eval_steps, save_steps, batch_size, num_epochs,
                                  logging_dir, logging_steps, seed, reporter, metric_for_best_model, greater_is_better,
                                  train_dataloader, valid_dataloader, tokenizer):

    model = FitnessRegressionModel(model_dir, model_src)
    if model_src == "official":
        load_best_model_at_end = False
    else:
        load_best_model_at_end = True

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
        load_best_model_at_end=load_best_model_at_end,
        ddp_find_unused_parameters=False,
        seed=seed,
        report_to=reporter,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
    )
    if model_src == "official":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=valid_dataloader.dataset,
            compute_metrics=compute_regression_metrics,
            callbacks=[SaveCompleteModelCallback(model=model, tokenizer=tokenizer)]
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=valid_dataloader.dataset,
            compute_metrics=compute_regression_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
                       SaveCompleteModelCallback(model=model, tokenizer=tokenizer)]
        )
    trainer.train()

    # Save the best model after training
    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    # 保存 PEFT (LoRA adapter)
    model.model.save_pretrained(best_model_path)

    # 保存分类头
    torch.save(model.regressor.state_dict(), f"{best_model_path}/regression_head.pt")

    # 保存 tokenizer（建议你提前加载并传入 tokenizer）
    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best trained regressor head [Fitness] saved to: {best_model_path}")

