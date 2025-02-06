# ------------------------------------------------------------------------------
# Author:    Dou zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/22
#
# Description: 
# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt


print('File to dataframe')
# 读取txt文件数据，假设文件名为data.txt，你可以根据实际情况替换文件名
file_path = r"/home/u2600215/qpacking/data/work/80_seq/80_resultDB.txt"
df = pd.read_csv(file_path, sep=' ', header=None, names=["col1", "col2", "value"])
# 定义区间
bins = [0, 0.25, 0.5, 0.75, 1.0]
labels = ['[0, 0.25]', '[0.25, 0.50]', '[0.50, 0.75]', '[0.75, 1.0]']

# 将 value 列的数据按照区间分组
df['value_group'] = pd.cut(df['value'], bins=bins, labels=labels, right=False)

# 计算每个区间的频数
frequency = df['value_group'].value_counts(sort=False)

# 计算百分比
percentage = (frequency / frequency.sum()) * 100

# 绘制频数分布直方图
plt.figure(figsize=(8, 6))
plt.bar(percentage.index, percentage.values, color='skyblue')

# 添加标签
plt.xlabel('Sequence identity', fontsize=16)
plt.ylabel('Percentage (%)', fontsize=16)
plt.title('80')

# 调整y轴的范围，确保柱子和文本不会超出边界
plt.ylim(0, max(percentage.values) * 1.1)  # 增加 20% 的空白空间，以确保文本显示

# 显示百分比值
for i, v in enumerate(percentage.values):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=14)

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=14)  # 横坐标字体大小
plt.yticks(fontsize=14)  # 纵坐标字体大小

print('Saving figure')
# 保存为高清TIF格式图片
plt.tight_layout()  # 自动调整子图参数，以给标签留出足够空间
output_path = r"/home/u2600215/qpacking/data/work/80_seq/heatmap.tif"
plt.savefig(output_path, format='tif', dpi=300)  # 设置 dpi=300 高分辨率
