import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def compute_tm_matrix(pdb_dir):
    """
    compute pair-wise TMscore matrix
    """
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    pdb_files.sort()
    n = len(pdb_files)

    tm_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            file1 = os.path.join(pdb_dir, pdb_files[i])
            file2 = os.path.join(pdb_dir, pdb_files[j])

            result = subprocess.run(["TMalign", file1, file2], capture_output=True, text=True)
            output = result.stdout

            for line in output.split("\n"):
                if line.startswith("TM-score="):
                    tm_score = float(line.split()[1])
                    tm_matrix[i, j] = tm_score
                    tm_matrix[j, i] = tm_score
                    break
    labels = [os.path.splitext(f)[0] for f in pdb_files]
    print(tm_matrix)
    return tm_matrix, labels


def plot_heatmap(M, labels, plot_path):
    M_plot = M.copy()

    M_plot[np.triu_indices_from(M_plot, k=0)] = np.nan

    plt.figure(figsize=(10, 8))

    cmap = plt.cm.coolwarm
    cmap.set_bad(color="white")

    plt.imshow(M_plot, cmap=cmap, aspect="auto")

    plt.colorbar(label="TM-score")
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Pairwise Sequence Identity Heatmap (Lower Triangle Only)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.show()


if __name__ == '__main__':
    pdb_dir = "/Users/douzhixin/Developer/qPacking-esm/data/structure/fitness"
    tm_matrix, pdb_files = compute_tm_matrix(pdb_dir)
    plot_heatmap(
        tm_matrix,
        pdb_files,
        plot_path=r"/Users/douzhixin/Developer/qPacking-esm/figure/python/identity/tmscore_heatmap.png",
    )
