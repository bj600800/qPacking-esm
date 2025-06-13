"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/5/1

# Description: 
# ------------------------------------------------------------------------------
"""
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

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
        print('error')
        return {}
    except Exception as e:
        print(e)
        return {}

def plot_area(data, title):
    # 转成 numpy 数组
    data = np.array(data)

    # 计算均值和标准差
    mu, sigma = np.mean(data), np.std(data)

    # 创建绘图
    plt.figure(figsize=(10, 6))

    # 绘制直方图
    sns.histplot(data, bins=10, stat="density", color='skyblue', edgecolor='black')

    # 单独绘制 KDE 曲线
    sns.kdeplot(data, color='steelblue', linewidth=2, label="KDE")
    # 高斯分布拟合曲线
    x = np.linspace(data.min(), data.max(), 1000)
    # x = np.linspace(data.min()-1000, data.max()+1000, 1000)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r--', label=f'Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}')

    # 标注与美化
    plt.title(f"Distribution of {title}")
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_degree(data, title):
    # 计算均值和标准差
    mu, sigma = np.mean(data), np.std(data)

    # 绘制直方图（归一化密度）
    count, bins, ignored = plt.hist(data, bins=range(min(data), max(data) + 2), density=True, alpha=0.6,
                                    color='skyblue', edgecolor='black')

    # 拟合的高斯分布曲线
    x = np.linspace(min(data) - 1, max(data) + 1, 1000)
    y = norm.pdf(x, mu, sigma)

    # 绘图
    plt.plot(x, y, 'r--', label=f'Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}')
    plt.title(f"Distribution of {title}")
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.show()

def plot_rsa(data, title):
    # 创建绘图
    plt.figure(figsize=(10, 6))

    # 绘制直方图并获取 bin 信息
    counts, bin_edges, _ = plt.hist(data, bins=10, density=True,
                                    color='skyblue', edgecolor='black', alpha=0.6)

    # 标注与美化
    plt.title(f"Distribution of {title}")
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.tight_layout()
    plt.show()

def split_feature(feature, key, data_type):
    single_feature = {}
    for k, v in feature.items():
        if key=='class':
            single_feature[k] = v[key]
        else:
            if data_type == 'float32':
                single_feature[k] = {key: np.float32(value) for key, value in v[key].items()}
            elif data_type == 'int':
                single_feature[k] = {key: value for key, value in v[key].items()}
    return single_feature



if __name__ == '__main__':
    from tqdm import tqdm
    output_pkl_file = r"/Users/douzhixin/Developer/qPacking/data/test/results.pkl"
    existing_results = load_existing_results(output_pkl_file)

    feature_names = {'class':'int', 'area': 'float32', 'degree': 'int', 'rsa': 'float32', 'order': 'float32', 'centrality': 'float32'}
    for name, data_type in tqdm(feature_names.items()):
        new_pkl = rf"/Users/douzhixin/Developer/qPacking/data/test/{name}_results.pkl"
        new_feature = split_feature(existing_results, name, data_type)
        with open(new_pkl, "wb") as f:
            pickle.dump(new_feature, f)

    # output_pkl_file = r"/Users/douzhixin/Developer/qPacking/data/feature/70/70_class_results.pkl"
    # existing_results = load_existing_results(output_pkl_file)
    #
    # for k, v in existing_results.items():
    #     for feature, value in v.items():
    #         print(f"{k} - {feature}: {value}")
    #     input()

    # dict_keys(['area', 'degree', 'rsa', 'order', 'centrality'])
    # area = [sum(v['area'].values()) for k, v in load_existing_results.items()]
    # plot_area(area, 'SASA Area')

    # degree = [i for k, v in load_existing_results.items() for i in v['degree'].values()]
    # plot_degree(degree, 'Packing Degree')
    # print(degree)

    # rsa = [i for k, v in load_existing_results.items() for i in v['rsa'].values()]
    # plot_rsa(rsa, 'rSAS')

    # order = [sum(v['order'].values()) for k, v in load_existing_results.items()]
    # plot_rsa(order, 'Packing order')
    #
    # centrality = [i for k, v in load_existing_results.items() for i in v['centrality'].values()]
    # plot_area(centrality, 'centrality')

