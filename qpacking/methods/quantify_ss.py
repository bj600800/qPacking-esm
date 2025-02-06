"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/12/12

# Description: Calculate orientation, length.
the longitudinal axis of a helix: alpha carbons of one helix indicate the axis
# ------------------------------------------------------------------------------
"""
import os

import biotite.structure.io as strucio
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def draw_3D(points, res_ids, residues, ss_id, protein_name, fig_dir):
    """
    To confirm the correctness of the computed regression line and direction vector.

    :param points:
    :param res_ids:
    :param residues:
    :param ss_id:
    :param protein_name:
    :param fig_dir:
    :return: figure HTML file
    """
    # 步骤1：中心化数据，关注趋势而非位置
    mean_coords = np.mean(points, axis=0)
    centered_coords = points - mean_coords

    # 使用PCA来计算回归直线
    pca = PCA(n_components=1)  # 主成分分析，n_components=1表示拟合一条直线
    pca.fit(centered_coords)

    # 主成分方向，即回归直线的方向
    line_direction = pca.components_[0]

    # 计算点云的均值（回归直线的起始点）
    line_start = np.mean(centered_coords, axis=0)

    # 计算点云中距离均值最远的点
    distances = np.linalg.norm(centered_coords - line_start, axis=1)
    max_distance_idx = np.argmax(distances)

    # 沿回归直线方向，生成两个端点，距离是点云中最远的距离
    t_min = -distances[max_distance_idx]  # 负方向上的端点
    t_max = distances[max_distance_idx]  # 正方向上的端点

    # 计算回归直线的两个端点
    line_end_1 = line_start + t_min * line_direction
    line_end_2 = line_start + t_max * line_direction

    # 计算回归直线的长度
    line_length = np.linalg.norm(line_end_2 - line_end_1)

    # 调整方向向量的长度，使其等于回归直线的长度
    normalized_direction = line_direction / np.linalg.norm(line_direction)  # 单位化方向向量
    direction_vector = normalized_direction * line_length / 2  # 调整方向向量的长度为回归直线的长度

    # 创建 Plotly 图形
    fig = go.Figure()

    # 绘制点云
    fig.add_trace(go.Scatter3d(
        x=centered_coords[:, 0], y=centered_coords[:, 1], z=centered_coords[:, 2],
        mode='markers+text', marker=dict(size=5, color='blue'),
        text=[f'{res_ids[i]}{residues[i]}' for i in range(len(res_ids))],
        textposition='top center', name='Points'
    ))

    # 绘制拟合的回归直线
    fig.add_trace(go.Scatter3d(
        x=[line_end_1[0], line_end_2[0]], y=[line_end_1[1], line_end_2[1]], z=[line_end_1[2], line_end_2[2]],
        mode='lines', line=dict(color='red', width=2),
        name='Regression Line'
    ))

    # 绘制回归直线的方向向量
    fig.add_trace(go.Cone(
        x=[line_start[0]], y=[line_start[1]], z=[line_start[2]],
        u=[direction_vector[0]], v=[direction_vector[1]], w=[direction_vector[2]],
        colorscale='Viridis', showscale=False, sizemode="scaled", sizeref=0.2
    ))

    # 计算数据的范围，以便调整坐标轴范围
    axis_min = np.array([centered_coords[:, 0].min(), centered_coords[:, 1].min(), centered_coords[:, 2].min()]).min()
    axis_max = np.array([centered_coords[:, 0].max(), centered_coords[:, 1].max(), centered_coords[:, 2].max()]).max()
    x_min, y_min, z_min = axis_min, axis_min, axis_min
    x_max, y_max, z_max = axis_max, axis_max, axis_max

    # 留出一点额外空间以避免点云贴着边缘
    margin = 1.1  # 用于扩展坐标轴的范围

    # 设置图形的显示范围和标签
    fig.update_layout(
        title=f'{protein_name}: {ss_id}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(
                range=[x_min - margin, x_max + margin],
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='black',  # 设置 x 轴为黑色线
                showticklabels=True,  # 显示刻度标签
                tickmode='auto',  # 自动生成刻度
                ticks='outside',  # 刻度标记显示在外部
                ticklen=5,  # 设置刻度线长度
            ),
            yaxis=dict(
                range=[y_min - margin, y_max + margin],
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='black',  # 设置 y 轴为黑色线
                showticklabels=True,  # 显示刻度标签
                tickmode='auto',  # 自动生成刻度
                ticks='outside',  # 刻度标记显示在外部
                ticklen=5,  # 设置刻度线长度
            ),
            zaxis=dict(
                range=[z_min - margin, z_max + margin],
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='black',  # 设置 z 轴为黑色线
                showticklabels=True,  # 显示刻度标签
                tickmode='auto',  # 自动生成刻度
                ticks='outside',  # 刻度标记显示在外部
                ticklen=5,  # 设置刻度线长度
            ),
            aspectmode="cube"  # 保证x、y、z轴比例相同
        ),
        margin=dict(l=10, r=10, b=10, t=40),  # 设置边距
        title_x=0.5,  # 设置图题居中
    )

    # 显示图形
    # fig.show()
    fig.write_html(os.path.join(fig_dir, f'{protein_name}_{ss_id}.html'))


def calc_axis(points):
    """
    calculate the axis of each secondary structure

    :param points:
    :return:
    orientation vector
    """

    # 中心化数据, 把重心平移到原点（0，0，0）：关注趋势而非位置
    mean_coords = np.mean(points, axis=0)
    centered_coords = points - mean_coords

    # 使用PCA来计算回归直线
    pca = PCA(n_components=1)  # 主成分分析，n_components=1表示拟合一条直线
    pca.fit(centered_coords)

    # 主成分方向，即回归直线的方向
    line_direction = pca.components_[0]

    return line_direction


def calc_length(points):
    point1 = points[0]
    point2 = points[-1]
    return np.linalg.norm(point2 - point1)


def run(protein_name, ss_data, structure_dir, plot_fig=False):
    ss_types = ['helix', 'strand']
    fig_dir = os.path.join(structure_dir, 'secondary_structure_figures')
    structure_path = os.path.join(structure_dir, protein_name + ".pdb")

    if os.path.exists(structure_path):
        structure = strucio.load_structure(structure_path)
        for s_type in ss_types:
            for ss_id, group in ss_data[s_type].items():
                ca_coords = []

                # for plotting figures
                res_ids = []
                residues = []

                for res in group['residues']:
                    ca_mask = (structure.res_id == res[0]) & \
                              (structure.atom_name == "CA")
                    coord = structure[ca_mask].coord[0]
                    ca_coords.append(coord)

                    # for plotting figures
                    if plot_fig:
                        res_ids.append(res[0])
                        residues.append(res[1])

                ca_coords = np.array(ca_coords)

                # start compute attributes
                orientation = calc_axis(ca_coords)
                length = calc_length(ca_coords)
                ss_data[s_type][ss_id]['features'] = {'orientation': orientation, 'length': length}

                # for plotting figures
                if plot_fig:
                    if not os.path.exists(fig_dir):
                        os.mkdir(fig_dir)
                    draw_3D(ca_coords, res_ids, residues, ss_id, protein_name, fig_dir)
    return ss_data
