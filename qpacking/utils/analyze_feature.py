"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/5/1

# Description: 
# ------------------------------------------------------------------------------
"""
import os
import pickle
from tqdm import tqdm
import numpy as np
from Bio import SeqIO

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from collections import Counter
from math import comb


from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

def load_existing_results(output_file):
    """
    Load existing results from a pickle file.
    :param output_file: existing results file
    :return: a dictionary containing loaded results
    """
    try:
        with open(output_file, "rb") as f:
            results_dict = pickle.load(f)  # output file only 1 obj.
            if not isinstance(results_dict, dict):
                return {}
            return results_dict
    except (FileNotFoundError, EOFError):
        logger.error('FileNotFoundError')
        return {}
    except Exception as e:
        logger.error(e)
        return {}
def analyze_class(load_existing_results):
    binary_positive_stats = []
    contrastive_same_cluster_stats = []
    for protein_id, feature in tqdm(load_existing_results.items()):
        # binary stats
        class_feature = feature['class']
        seq_length = len(feature['area'])  # 'area' contains features of all residues
        hydrophobic_residue_num = len(class_feature)
        positive_hydrophobic_ratio = hydrophobic_residue_num / seq_length
        binary_positive_stats.append(positive_hydrophobic_ratio)

        # contrastive stats
        counts = Counter(class_feature.values())
        total_pairs = comb(len(class_feature), 2)  # n(p,k) >= 3
        same_pairs = sum(comb(size, 2) for size in counts.values())
        same_cluster_ratio = same_pairs / total_pairs
        contrastive_same_cluster_stats.append(same_cluster_ratio)

    mean_binary_positive_stats = np.mean(binary_positive_stats)
    mean_same_cluster_ratio = np.mean(contrastive_same_cluster_stats)
    print('mean_binary_positive_stats: ', mean_binary_positive_stats)
    print('mean_binary_negative_stats: ', str(1 - mean_binary_positive_stats))
    print()
    print('mean_same_cluster_ratio: ', mean_same_cluster_ratio)
    print('mean_different_cluster_ratio: ', str(1 - mean_same_cluster_ratio))

def plot_distribution_analysis(values, title_prefix="Feature", bins=20, epsilon=1e-6):
    """
    可视化任意一组连续值的分布，包括原始、log变换、Z-score标准化分布及对应的Q-Q图。

    Args:
        values (list or np.ndarray): 输入的一维连续变量。
        title_prefix (str): 图标题前缀，默认是"Feature"。
        bins (int): 直方图的柱数。
        epsilon (float): 避免 log(0) 的平滑项。
    """
    values = np.array(values)
    log_values = np.log(values + epsilon)
    zscore_values = (values - np.mean(values)) / np.std(values)

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # 原始直方图
    axs[0, 0].hist(values, bins=bins, color='skyblue', edgecolor='black')
    axs[0, 0].set_title(f"{title_prefix} - Original Histogram")

    # 原始 Q-Q 图
    stats.probplot(values, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title(f"{title_prefix} - Original Q-Q Plot")

    # 对数直方图
    axs[1, 0].hist(log_values, bins=bins, color='lightgreen', edgecolor='black')
    axs[1, 0].set_title(f"{title_prefix} - Log-Transformed Histogram")

    # 对数 Q-Q 图
    stats.probplot(log_values, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title(f"{title_prefix} - Log-Transformed Q-Q Plot")

    # Z-score 直方图
    axs[2, 0].hist(zscore_values, bins=bins, color='salmon', edgecolor='black')
    axs[2, 0].set_title(f"{title_prefix} - Z-score Histogram")

    # Z-score Q-Q 图
    stats.probplot(zscore_values, dist="norm", plot=axs[2, 1])
    axs[2, 1].set_title(f"{title_prefix} - Z-score Q-Q Plot")

    plt.tight_layout()
    plt.show()

def plot_regression(data, title):
    data = np.array(data)

    # Calculate mean and sd
    mu, sigma = np.mean(data), np.std(data)

    # 进行 Z-score 标准化
    data = (data - mu) / sigma

    plt.figure(figsize=(10, 6))

    # plot histogram
    sns.histplot(data, bins=5, stat="density", color='skyblue', edgecolor='black')

    # plot kde curve
    sns.kdeplot(data, color='steelblue', linewidth=2, label="KDE")

    # fit standard normal Gaussian curve
    x = np.linspace(data.min(), data.max(), 1000)
    y = norm.pdf(x, 0, 1)  # 因为此时已经标准化，所以均值为 0，std 为 1
    plt.plot(x, y, 'r--', label=f'Gaussian Fit\nμ=0.00, σ=1.00')

    plt.title(f"Distribution of {title}")
    plt.xlabel("Z-score Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)

    plt.legend()
    plt.tight_layout()
    plt.show()

def split_feature(feature, key, data_type):
    single_feature = {}
    for k, v in feature.items():
        if key=='class':
            single_feature[k] = v[key]
        else:
            if data_type == 'float32':
                single_feature[k] = {key: float(value) for key, value in v[key].items()}
            elif data_type == 'int':
                single_feature[k] = {key: value for key, value in v[key].items()}
    return single_feature

def run_split(input_pkl):
    dir_path = os.path.dirname(input_pkl)
    file_name = os.path.basename(input_pkl)
    existing_results = load_existing_results(input_pkl)
    feature_names = {'class': 'int', 'area': 'float32', 'degree': 'int', 'rsa': 'float32', 'order': 'float32', 'centrality': 'float32'}
    for name, data_type in tqdm(feature_names.items()):
        new_pkl = os.path.join(dir_path, file_name.split('.')[0]+f'_{name}.pkl')
        new_feature = split_feature(existing_results, name, data_type)
        with open(new_pkl, "wb") as f:
            pickle.dump(new_feature, f)
    # existing_results = load_existing_results(r"/Users/douzhixin/Developer/qPacking/data/test/results_centrality.pkl")
    # print(existing_results)

def prepare_plot_feature(load_existing_results, feature_name='centrality', bins=10):
    data = [i for protein_id, feature in load_existing_results.items() for i in feature[feature_name].values()]
    counts_density, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_widths = np.diff(bin_edges)  # 计算每个bin的宽度
    print(f"{feature_name} feature")
    # 计算bin中心
    bin_centers = bin_edges[:-1] + bin_widths / 2
    print("bin_centers:")
    for i in bin_centers:
        print(i)
    print()
    # 频率 = 频率密度 * bin宽度，保证频率总和为1
    frequencies = counts_density * bin_widths
    print("frequencies:")
    for i in frequencies:
        print(i)

    # # 验证频率之和为1
    # print("Sum of frequencies:", frequencies.sum())







if __name__ == '__main__':
    pkl_file = r"/Users/douzhixin/Developer/qPacking/data/feature/80/80_results.pkl"
    fasta_path = r"/Users/douzhixin/Developer/qPacking/data/sequence/complete_tim_80.fasta"
    data = load_existing_results(pkl_file)
    prepare_plot_feature(data)





    # analyze_class(data)
    # run_split(pkl_file)

    # data = load_existing_results(degree_pkl)
    # print(data)
    # input()
    # degree_data = [d for p, f in data.items() for i, d in f.items()]
    # plot_degree(degree_data)