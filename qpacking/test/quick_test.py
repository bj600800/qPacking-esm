import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from transformers import EsmTokenizer, EsmModel
from peft import PeftModel, PeftConfig


def mutate(seq, pos, mut_aa):
    """
    生成突变序列（基于0-index）
    """
    return seq[:pos] + mut_aa + seq[pos+1:]


def get_cls_embedding(model, tokenizer, sequence):
    """
    提取给定序列的 CLS token embedding（最后一层）
    """
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    cls_emb = outputs.hidden_states[-1][0, 0]  # shape: (embedding_dim,)
    return cls_emb.cpu().numpy()


def compute_best_dim_correlation(embeddings, labels):
    """
    计算 embedding 每个维度与 label 的 Pearson/Spearman 相关性，返回最大值
    """
    pearsons = [pearsonr(embeddings[:, i], labels)[0] for i in range(embeddings.shape[1])]
    spearmans = [spearmanr(embeddings[:, i], labels)[0] for i in range(embeddings.shape[1])]
    return max(pearsons), max(spearmans)


def main():
    # === 配置路径和序列 ===
    csv_path = "/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/ss.csv"
    best_model_path = "/checkpoints/bak/hydrophobic_contrastive_80_150/best"
    wt_seq = "NITAIIAEYKRKSPSGLDVERDPIEYSKFMERYAVGLSILTEEKYFNGSYETLRKIASSVSIPILMKDFIVKESQIDDAYNLGADTVLLIVKILTERELESLLEYARSYGMEPLIEINDENDLDIALRIGARFIGINSRDLETLEINKENQRKLISMIPSNVVKVAESGISERNEIEELRKLGVNAFLIGSSLMRNPEKIKEFIL"

    # === 读取突变数据 ===
    df = pd.read_csv(csv_path, header=0)
    df.columns = ["id", "pos", "wt", "mut", "fitness"]

    # === 加载 tokenizer 和模型 ===
    print("🔧 加载模型与Tokenizer...")
    peft_config = PeftConfig.from_pretrained(best_model_path)
    tokenizer = EsmTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # 未微调的 ESM 模型
    model_base = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
    model_base.eval()

    # 微调后的模型（PEFT LoRA）
    model_tuned = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
    model_tuned = PeftModel.from_pretrained(model_tuned, best_model_path)
    model_tuned.eval()

    # === 提取 embedding ===
    print("🔍 提取突变序列 embedding...")
    embeds_base = []
    embeds_tuned = []
    labels = []

    for _, row in df.iterrows():
        pos = int(row["pos"]) - 44  # 转为0-index
        mut_aa = row["mut"]
        mut_seq = mutate(wt_seq, pos, mut_aa)

        emb_base = get_cls_embedding(model_base, tokenizer, mut_seq)
        emb_tuned = get_cls_embedding(model_tuned, tokenizer, mut_seq)

        embeds_base.append(emb_base)
        embeds_tuned.append(emb_tuned)
        labels.append(row["fitness"])

    embeds_base = np.vstack(embeds_base)
    embeds_tuned = np.vstack(embeds_tuned)
    labels = np.array(labels)

    # === 分析相关性 ===
    print("📈 分析 embedding 与 fitness 的相关性...")
    p_base, s_base = compute_best_dim_correlation(embeds_base, labels)
    p_tuned, s_tuned = compute_best_dim_correlation(embeds_tuned, labels)

    print("\n=== 📊 结果比较 ===")
    print(f"未微调模型:     Pearson = {p_base:.3f}, Spearman = {s_base:.3f}")
    print(f"微调后模型:     Pearson = {p_tuned:.3f}, Spearman = {s_tuned:.3f}")

    if p_tuned > p_base and s_tuned > s_base:
        print("\n✅ 结论：结构特征微调提升了模型的fitness表示能力！")
    else:
        print("\n⚠️ 结论：结构特征微调未带来明显提升。请检查特征或微调方式。")


if __name__ == "__main__":
    main()

"""
binary
未微调模型:     Pearson = 0.240, Spearman = 0.348
微调后模型:     Pearson = 0.226, Spearman = 0.269

degree
未微调模型:     Pearson = 0.240, Spearman = 0.348
微调后模型:     Pearson = 0.261, Spearman = 0.328

area
未微调模型:     Pearson = 0.240, Spearman = 0.348
微调后模型:     Pearson = 0.291, Spearman = 0.280

order
未微调模型:     Pearson = 0.240, Spearman = 0.348
微调后模型:     Pearson = 0.203, Spearman = 0.264

rsa
未微调模型:     Pearson = 0.240, Spearman = 0.348
微调后模型:     Pearson = 0.199, Spearman = 0.247

centrality
未微调模型:     Pearson = 0.240, Spearman = 0.348
微调后模型:     Pearson = 0.252, Spearman = 0.250

contrastive
未微调模型:     Pearson = 0.240, Spearman = 0.348
微调后模型:     Pearson = 0.216, Spearman = 0.261
"""