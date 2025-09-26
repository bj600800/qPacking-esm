# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/7/6

# Description: This task prediction is noised
# ------------------------------------------------------------------------------
import torch
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO
)


def extract_embeddings_with_protein_ids(model, dataloader, device, save_dir, prefix="train"):
    """
    提取embedding，同时用batch中行索引作为蛋白ID

    Args:
        model, dataloader, device, save_dir, prefix 同前

    Returns:
        emb_path, lab_path, pid_path
    """
    model.eval()
    all_embeddings, all_labels, all_protein_ids = [], [], []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {prefix} embeddings"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)

            embeddings = outputs.last_hidden_state.cpu().numpy()  # [B, L, D]
            labels = batch['labels'].cpu().numpy()  # [B, L]

            B, L = labels.shape
            embeddings = embeddings.reshape(B * L, -1)
            labels = labels.reshape(B * L)

            # 构造蛋白ID: 每一行的所有token都对应行号作为蛋白ID
            protein_ids = np.repeat(np.arange(B), L)

            mask = labels != -100
            all_embeddings.append(embeddings[mask])
            all_labels.append(labels[mask])
            all_protein_ids.append(protein_ids[mask])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_protein_ids = np.concatenate(all_protein_ids, axis=0)

    emb_path = os.path.join(save_dir, f"{prefix}_embeddings.npy")
    lab_path = os.path.join(save_dir, f"{prefix}_labels.npy")
    pid_path = os.path.join(save_dir, f"{prefix}_protein_ids.npy")

    np.save(emb_path, all_embeddings)
    np.save(lab_path, all_labels)
    np.save(pid_path, all_protein_ids)

    logger.info(f"Saved {prefix} embeddings to {emb_path}, shape: {all_embeddings.shape}")
    logger.info(f"Saved {prefix} labels to {lab_path}, shape: {all_labels.shape}")
    logger.info(f"Saved {prefix} protein IDs to {pid_path}, shape: {all_protein_ids.shape}")

    return emb_path, lab_path, pid_path


def cluster_accuracy(true_labels, pred_labels):
    """
    Hungarian算法对预测标签重新编号，计算准确率

    Args:
        true_labels: np.ndarray
        pred_labels: np.ndarray

    Returns:
        accuracy: float
        new_preds: np.ndarray
    """
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    new_preds = np.array([mapping.get(p, p) for p in pred_labels])
    acc = (new_preds == true_labels).sum() / len(true_labels)
    return acc, new_preds


def evaluate_per_protein(true_labels, pred_labels, protein_ids):
    """
    按蛋白分组计算ARI、NMI、Aligned Accuracy的均值

    Args:
        true_labels: np.ndarray [N]
        pred_labels: np.ndarray [N]
        protein_ids: np.ndarray [N]

    Returns:
        dict: {'mean_ari', 'mean_nmi', 'mean_aligned_acc', 'protein_count'}
    """
    unique_proteins = np.unique(protein_ids)
    ari_list, nmi_list, aligned_acc_list = [], [], []

    for pid in unique_proteins:
        mask = protein_ids == pid
        true_sub = true_labels[mask]
        pred_sub = pred_labels[mask]

        # 过滤无效标签
        valid_mask = true_sub != -100
        true_sub = true_sub[valid_mask]
        pred_sub = pred_sub[valid_mask]

        if len(true_sub) == 0:
            continue

        ari = adjusted_rand_score(true_sub, pred_sub)
        nmi = normalized_mutual_info_score(true_sub, pred_sub)
        acc, _ = cluster_accuracy(true_sub, pred_sub)

        ari_list.append(ari)
        nmi_list.append(nmi)
        aligned_acc_list.append(acc)

    return {
        "mean_ari": np.mean(ari_list) if ari_list else 0,
        "mean_nmi": np.mean(nmi_list) if nmi_list else 0,
        "mean_aligned_acc": np.mean(aligned_acc_list) if aligned_acc_list else 0,
        "protein_count": len(ari_list)
    }


def knn_predict(eval_embeddings, train_embeddings, train_labels, k=1):
    """
    预测eval_embeddings的簇标签，用kNN
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_embeddings, train_labels)
    preds = knn.predict(eval_embeddings)
    return preds


def extract_and_evaluate_per_protein(model, dataloader, device, save_dir, train_emb_path, train_lab_path, prefix="valid", k=1):
    """
    提取embedding + kNN预测 + 蛋白分组评估

    Returns:
        dict：蛋白级聚类评估指标
    """
    emb_path, lab_path, pid_path = extract_embeddings_with_protein_ids(model, dataloader, device, save_dir, prefix)
    eval_embeddings = np.load(emb_path)
    eval_labels = np.load(lab_path)
    eval_protein_ids = np.load(pid_path)

    train_embeddings = np.load(train_emb_path)
    train_labels = np.load(train_lab_path)

    preds = knn_predict(eval_embeddings, train_embeddings, train_labels, k=k)

    results = evaluate_per_protein(eval_labels, preds, eval_protein_ids)

    logger.info(f"Protein-level evaluation on {results['protein_count']} proteins:")
    logger.info(f"Mean ARI: {results['mean_ari']:.4f}, Mean NMI: {results['mean_nmi']:.4f}, Mean Aligned Accuracy: {results['mean_aligned_acc']:.4f}")

    print(f"Protein-level mean ARI: {results['mean_ari']:.4f}")
    print(f"Protein-level mean NMI: {results['mean_nmi']:.4f}")
    print(f"Protein-level mean Aligned Accuracy: {results['mean_aligned_acc']:.4f}")

    return results


# -----------------------------------------
# 你原来的extract_embeddings可改为extract_embeddings_with_protein_ids，
# 你需要确保你的dataset返回batch包含 'protein_id' 张量，表示每个token的蛋白ID（整数）
# -----------------------------------------

# 使用示例：
if __name__ == "__main__":
    from transformers import EsmModel, EsmTokenizer
    from peft import PeftModel
    from qpacking.data import dataset

    fasta_file = r"/Users/douzhixin/Developer/qPacking/data/test/sequence.fasta"
    pkl_file = r"/Users/douzhixin/Developer/qPacking/data/test/results_class.pkl"
    model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/qpacking/hydrophobic_contrastive/checkpoint-1"
    tokenized_cache_path = r"/Users/douzhixin/Developer/qPacking/data/test/tokenized_cache"
    task = r"hydrophobic_contrastive"
    test_ratio = 0.1
    batch_size = 16
    seed = 4309

    base_model_name = "/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t30_150M_UR50D"
    base_model = EsmModel.from_pretrained(base_model_name)
    tokenizer = EsmTokenizer.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader, valid_dataloader, _ = dataset.run_structure_encoder(
        fasta_file, pkl_file, model_dir, tokenized_cache_path, task, test_ratio, batch_size, seed)

    # 提取训练集embedding，注意此函数也要改为extract_embeddings_with_protein_ids，如果需要蛋白id可存储
    train_emb_path, train_lab_path, train_pid_path = extract_embeddings_with_protein_ids(
        model, train_dataloader, device, model_dir, prefix="train")

    # 验证集蛋白分组评估
    eval_results = extract_and_evaluate_per_protein(
        model, valid_dataloader, device, model_dir,
        train_emb_path, train_lab_path, prefix="valid", k=1)

    print(f"Protein-level mean Aligned Accuracy: {eval_results['mean_aligned_acc']:.4f}")
