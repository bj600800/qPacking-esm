"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/5

# Description: setup trainer for fine-tuning esm-2
# ------------------------------------------------------------------------------
"""
import numpy as np
import evaluate
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import EsmTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback

from qpacking.models import dataset
from qpacking.models.model import TokenClassificationModel


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # 过滤掉 -100 的位置
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

    accuracy_metric = evaluate.load("accuracy")
    acc = accuracy_metric.compute(predictions=true_predictions, references=true_labels)["accuracy"]

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }


def train_cluster_classification(model_dir, checkpoints_dir, logging_dir,
          lora_rank, lora_alpha, lora_dropout,
          batch_size, num_epochs, seed, lr, num_clusters,
          train_dataloader, valid_dataloader):

    # load model
    model = TokenClassificationModel(
        model_dir=model_dir,
        num_clusters=num_clusters,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    # load tokenizer
    # tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)

    # define training arguments
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=logging_dir,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),  # 如果支持，自动混合精度
    )

    # setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=valid_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # train!
    trainer.train()

if __name__ == '__main__':
    model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t30_150M_UR50D"
    checkpoints_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/qpacking"
    logging_dir = r"/Users/douzhixin/Developer/qPacking/code/logs"

    fasta_file = r"/Users/douzhixin/Developer/qPacking/data/test/sequence.fasta"
    pkl_file = r"/Users/douzhixin/Developer/qPacking/data/test/class_results.pkl"
    lr = 2e-5
    lora_rank = 8
    lora_alpha = 8
    lora_dropout = 0.05
    batch_size = 8
    num_epochs = 500
    seed = 3407
    test_ratio = 0.1
    train_dataloader, valid_dataloader, num_clusters = dataset.run(fasta_file, pkl_file, model_dir, test_ratio, batch_size, seed)

    train_cluster_classification(model_dir, checkpoints_dir, logging_dir,
          lora_rank, lora_alpha, lora_dropout,
          batch_size, num_epochs, seed, lr, num_clusters,
          train_dataloader, valid_dataloader)