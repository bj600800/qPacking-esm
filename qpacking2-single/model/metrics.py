"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/30

# Description: 
# ------------------------------------------------------------------------------
"""
import os
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from train_configs import Config

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
    Args:
        eval_pred:
    Returns:
        dict:
    """
    predictions, labels = eval_pred
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)

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

def compute_raw_regression_metrics(pred, yaml_path, task):
    preds = pred.predictions.squeeze()  # shape: (batch,)
    labels = pred.label_ids.squeeze()
    config = Config.from_yaml(yaml_path, task)
    mu_sigma_path = os.path.join(config.path.tokenized_cache_path, "mu_sigma.json")
    with open(mu_sigma_path, "r") as f:
        stats = json.load(f)
    mu = stats["mu"]
    sigma = stats["sigma"]

    preds_raw = preds * sigma + mu
    labels_raw = labels * sigma + mu

    return {
        "pearsonr": pearsonr(preds_raw, labels_raw).statistic,
        "spearmanr": spearmanr(preds_raw, labels_raw).statistic,
        "mse_raw": mean_squared_error(preds_raw, labels_raw)
    }