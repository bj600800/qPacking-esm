"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/30
#
# Description:
#   Read FASTA sequences, compute pairwise identity using global alignment,
#   and output a heatmap.
# ------------------------------------------------------------------------------
"""

from Bio import SeqIO, pairwise2
import numpy as np
import matplotlib.pyplot as plt
from Bio.pairwise2 import format_alignment


def seq_identity(s1, s2):
    """
    Compute global alignment (no scoring matrix)
    and return identity = matched positions / aligned length.
    """
    aln = pairwise2.align.globalxx(s1, s2, one_alignment_only=True)[0]
    aligned1, aligned2 = aln.seqA, aln.seqB
    matches = sum(a == b for a, b in zip(aligned1, aligned2))
    return matches / len(aligned1)


def compute_identity_matrix(seqs):
    """Return NxN identity matrix for sequences list."""
    n = len(seqs)
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                M[i, j] = seq_identity(seqs[i], seqs[j])
            else:
                M[i, j] = M[j, i]
    return M


def plot_heatmap(M, labels, plot_path):
    # 复制矩阵避免修改原矩阵
    M_plot = M.copy()

    # 去掉对角线和上半部分
    M_plot[np.triu_indices_from(M_plot, k=0)] = np.nan

    plt.figure(figsize=(10, 8))

    # NaN 设为白色
    cmap = plt.cm.Blues
    cmap.set_bad(color="white")

    # 不强制正方形显示
    plt.imshow(M_plot, cmap=cmap, aspect="auto")

    plt.colorbar(label="Sequence Identity")
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Pairwise Sequence Identity Heatmap (Lower Triangle Only)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.show()


# --------------------------
# Main
# --------------------------
# 输入 FASTA 文件路径
fasta_file = "/Users/douzhixin/Developer/qPacking-esm/data/sequence/fitness.fasta"
plot_path = r"/Users/douzhixin/Developer/qPacking-esm/figure/python/identity/sequence_identity_heatmap.png"
# 读取所有序列
records = list(SeqIO.parse(fasta_file, "fasta"))
seqs = [str(r.seq) for r in records]
labels = [r.id for r in records]

# 计算 Identity 矩阵
identity_matrix = compute_identity_matrix(seqs)

# 画热图
plot_heatmap(identity_matrix, labels, plot_path)
