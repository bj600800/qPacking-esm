"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/5

# Description: setup trainer for fine-tuning esm-2
# ------------------------------------------------------------------------------
"""
from typing import Optional

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
import mlflow
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)

from qpacking.models.model import TokenClassificationModel
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    """
    Focal Loss: https://arxiv.org/pdf/1708.02002
    """
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(weight=self.alpha,
                                   reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = x.device  # 保证所有计算都在 x 的 device 上

        if self.alpha is not None and self.alpha.device != device:
            self.alpha = self.alpha.to(device)
            self.nll_loss.weight = self.alpha  # 更新 NLLLoss 内部的 weight

        if x.ndim > 2:
            batch_size, seq_len, num_classes = x.shape
            x = x.view(-1, num_classes)  # (B * L, C)
            y = y.view(-1)               # (B * L,)

        y = y.to(device)

        mask = y != self.ignore_index
        x, y = x[mask], y[mask]
        if len(y) == 0:
            return torch.tensor(0.0, device=device)

        log_p = F.log_softmax(x, dim=-1)
        ce_loss = self.nll_loss(log_p, y)

        log_pt = log_p[torch.arange(len(y), device=device), y]
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

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

def compute_metrics(eval_preds):
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


def train_cluster_classification(
        model_dir, checkpoints_dir, logging_dir,
        batch_size, num_epochs, seed, lr,
        train_dataloader, valid_dataloader,
        lora_rank, lora_alpha, lora_dropout,
        eval_steps, save_steps, eval_strategy, save_strategy,logging_strategy,
        logging_steps, save_total_limit,
        reporter, metric_for_best_model, greater_is_better):

    model = TokenClassificationModel(
        model_dir=model_dir,
        num_clusters=2,  # binary class 1/0
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        seed=seed,
        report_to=reporter,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available()
    )


    # trainer = FocalLossTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataloader.dataset,
    #     eval_dataset=valid_dataloader.dataset,
    #     data_collator=train_dataloader.collate_fn,
    #     compute_metrics=compute_metrics,
    #     focal_gamma=focal_gamma,
    #     focal_alpha=focal_alpha,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=valid_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()

    # save the best
    best_model_path = f"{checkpoints_dir}/best"
    # 如果是 LoRA 微调模型（PeftModel），用 save_pretrained 正确保存 adapter
    if isinstance(trainer.model.model, PeftModel):
        trainer.model.model.save_pretrained(best_model_path)
        logger.info(f"The best fine-tuned model saved to: {best_model_path}")
    else:
        logger.warning("The model is not a PeftModel. Skipping save_pretrained.")

