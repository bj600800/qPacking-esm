# ------------------------------------------------------------------------------
# Author:    Dou zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/22
#
# Description: 
# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

file_path = r"/home/u2600215/qpacking/data/work/80_seq/80_resultDB.txt"
df = pd.read_csv(file_path, sep=' ', header=None, names=["col1", "col2", "value"])
bins = [0, 0.25, 0.5, 0.75, 1.0]
# x label names
labels = ['[0, 0.25]', '[0.25, 0.50]', '[0.50, 0.75]', '[0.75, 1.0]']

# divided values depend on bins.
df['value_group'] = pd.cut(df['value'], bins=bins, labels=labels, right=False)

frequency = df['value_group'].value_counts(sort=False)
percentage = (frequency / frequency.sum()) * 100

plt.figure(figsize=(8, 6))
plt.bar(percentage.index, percentage.values, color='skyblue')

plt.xlabel('Sequence identity', fontsize=16)
plt.ylabel('Percentage (%)', fontsize=16)
plt.title('80')
plt.ylim(0, max(percentage.values) * 1.1)  # 增加 20% 的空白空间，以确保文本显示

# show percentage number on the plot bars
for i, v in enumerate(percentage.values):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()  # layout style
output_path = r"/home/u2600215/qpacking/data/work/80_seq/heatmap.tif"
plt.savefig(output_path, format='tif', dpi=300)
logger.info(f"output image saved to {output_path}")