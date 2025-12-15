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


def seq_identity(s1, s2):
    aln = pairwise2.align.globalxx(s1, s2, one_alignment_only=True)[0]
    aligned1, aligned2 = aln.seqA, aln.seqB
    matches = sum(a == b for a, b in zip(aligned1, aligned2))
    return matches / len(aligned1)


def compute_identity_matrix(seqs):
    """Return NxN identity matrix for the sequences list input."""
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
    M_plot = M.copy()

    M_plot[np.triu_indices_from(M_plot, k=0)] = np.nan  # keep the half one

    plt.figure(figsize=(10, 8))

    cmap = plt.cm.Blues
    cmap.set_bad(color="white")

    plt.imshow(M_plot, cmap=cmap, aspect="auto")

    plt.colorbar(label="Sequence Identity")
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Pairwise Sequence Identity Heatmap (Lower Triangle Only)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.show()

if __name__ == '__main__':

    fasta_file = "/Users/douzhixin/Developer/qPacking-esm/data/sequence/fitness.fasta"
    plot_path = r"/Users/douzhixin/Developer/qPacking-esm/figure/python/identity/sequence_identity_heatmap.png"

    records = list(SeqIO.parse(fasta_file, "fasta"))
    seqs = [str(r.seq) for r in records]
    labels = [r.id for r in records]

    identity_matrix = compute_identity_matrix(seqs)
    plot_heatmap(identity_matrix, labels, plot_path)
