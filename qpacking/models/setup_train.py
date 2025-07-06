"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/5

# Description: setup trainer for fine-tuning esm-2
# ------------------------------------------------------------------------------
"""


import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
)

from qpacking.models.model import (HydrophobicBinaryClassificationModel, FocalLoss,
                                   HydrophobicContrastiveModel, tokenwise_supervised_contrastive_loss_batch)
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


def train_hydrophobic_binary_classification(
        model_dir, checkpoints_dir, logging_dir,
        batch_size, num_epochs, seed, lr, num_clusters,
        train_dataloader, valid_dataloader,
        lora_rank, lora_alpha, lora_dropout,
        eval_steps, save_steps, eval_strategy, save_strategy,logging_strategy,
        logging_steps, save_total_limit,
        reporter, metric_for_best_model, greater_is_better):

    model = HydrophobicBinaryClassificationModel(
        model_dir=model_dir,
        num_clusters=num_clusters,  # binary class: 1/0
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
        compute_metrics=compute_binary_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()

    # Save the best model after training
    best_model_path = f"{checkpoints_dir}/best"
    if isinstance(trainer.model, PeftModel):
        trainer.model.save_pretrained(best_model_path)
        logger.info(f"The best fine-tuned PEFT model saved to: {best_model_path}")
    else:
        trainer.save_model(best_model_path)
        logger.info(f"The best fine-tuned model saved to: {best_model_path}")


def train_hydrophobic_contrastive_model(
        model_dir, lora_rank, lora_alpha, lora_dropout, proj_dim,
        checkpoints_dir, lr, eval_strategy, save_strategy, logging_strategy,
        eval_steps, save_steps, batch_size, num_epochs, logging_dir, logging_steps,
        save_total_limit, seed, reporter, metric_for_best_model, greater_is_better,
        train_dataloader, valid_dataloader):
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=valid_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save the best model after training
    best_model_path = f"{checkpoints_dir}/best"
    if isinstance(trainer.model, PeftModel):
        trainer.model.save_pretrained(best_model_path)
        logger.info(f"The best fine-tuned PEFT model saved to: {best_model_path}")
    else:
        trainer.save_model(best_model_path)
        logger.info(f"The best fine-tuned model saved to: {best_model_path}")


