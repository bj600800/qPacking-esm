"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/3/12

# Description: converting batch unstructured hydrocluster data into numerical vectors for model training.
# 1. using vectorizer.py to convert raw data into numerical vectors (Appropriate embedding methods).
# 2. save vectorization to pickle file  -> features.pkl
# ------------------------------------------------------------------------------
"""
from qpacking.hydrocluster.cluster_analyzer import Analyzer  # calculate metrics of hydrophobic clusters
