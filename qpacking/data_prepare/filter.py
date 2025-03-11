"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/12/12

# Description: filter the incomplete TIM fold structures from complete ones.
# without the massive multiple structure alignments.

detect beta-barrel and alpha-barrel completeness

# if passed the fold filter modules, return true, else return false.
# ------------------------------------------------------------------------------
"""

import os

import networkx as nx
from pyvis.network import Network
import itertools

import numpy as np
import plotly.graph_objects as go

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

def draw_ss_3D(points, group, ss_id, protein_name, fig_dir):
    """
    draw all ss in a figure.
    To confirm the correctness of the computed regression line and direction vector.

    :param points: {ss_id:[[x,y,z], [x,y,z]], ss_id:[[x,y,z], [x,y,z]]}
    :param group: {'residues': [[24, 'P', 'H', [0, 4]], [25, 'E', 'H', [2, 4]]}
    :param ss_id: helix_1
    :param protein_name: AF-A0A009EPY2-F1-model_v4_TED01
    :param fig_dir: directory to save figure
    :return: HTML file
    """

    res_ids = [res[0] for res in group]
    residues = [res[1] for res in group]

    # move points to zero
    mean_coords = np.mean(points, axis=0)
    centered_coords = points - mean_coords

    # using PCA to calculate the vector
    pca = PCA(n_components=1)  # n_components=1 means line fitting
    pca.fit(centered_coords)

    # regression line direction
    line_direction = pca.components_[0]

    line_start = np.mean(centered_coords, axis=0)

    # calculate the distance between the points and the start point of the line
    distances = np.linalg.norm(centered_coords - line_start, axis=1)
    max_distance_idx = np.argmax(distances)

    # generate the two endpoints of the regression line
    t_min = -distances[max_distance_idx]
    t_max = distances[max_distance_idx]

    # calculate the two endpoints of the regression line
    line_end_1 = line_start + t_min * line_direction
    line_end_2 = line_start + t_max * line_direction

    # calculate the length of the regression line
    line_length = np.linalg.norm(line_end_2 - line_end_1)

    # modify the direction vector to the length of the regression line
    normalized_direction = line_direction / np.linalg.norm(line_direction)  # 单位化方向向量
    direction_vector = normalized_direction * line_length / 2  # 调整方向向量的长度为回归直线的长度

    # create a plotly figure
    fig = go.Figure()

    # draw
    fig.add_trace(go.Scatter3d(
        x=centered_coords[:, 0], y=centered_coords[:, 1], z=centered_coords[:, 2],
        mode='markers+text', marker=dict(size=5, color='blue'),
        text=[f'{res_ids[i]}{residues[i]}' for i in range(len(res_ids))],
        textposition='top center', name='Points'
    ))

    # draw line
    fig.add_trace(go.Scatter3d(
        x=[line_end_1[0], line_end_2[0]], y=[line_end_1[1], line_end_2[1]], z=[line_end_1[2], line_end_2[2]],
        mode='lines', line=dict(color='red', width=2),
        name='Regression Line'
    ))

    # draw direction
    fig.add_trace(go.Cone(
        x=[line_start[0]], y=[line_start[1]], z=[line_start[2]],
        u=[direction_vector[0]], v=[direction_vector[1]], w=[direction_vector[2]],
        colorscale='Viridis', showscale=False, sizemode="scaled", sizeref=0.2
    ))

    # calc range of data
    axis_min = np.array([centered_coords[:, 0].min(), centered_coords[:, 1].min(), centered_coords[:, 2].min()]).min()
    axis_max = np.array([centered_coords[:, 0].max(), centered_coords[:, 1].max(), centered_coords[:, 2].max()]).max()
    x_min, y_min, z_min = axis_min, axis_min, axis_min
    x_max, y_max, z_max = axis_max, axis_max, axis_max

    # layout
    margin = 1.1

    # setup labels
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
                linecolor='black',
                showticklabels=True,
                tickmode='auto',
                ticks='outside',
                ticklen=5,
            ),
            yaxis=dict(
                range=[y_min - margin, y_max + margin],
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='black',
                showticklabels=True,
                tickmode='auto',
                ticks='outside',
                ticklen=5,
            ),
            zaxis=dict(
                range=[z_min - margin, z_max + margin],
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='black',
                showticklabels=True,
                tickmode='auto',
                ticks='outside',
                ticklen=5,
            ),
            aspectmode="cube"
        ),
        margin=dict(l=10, r=10, b=10, t=40),  # 设置边距
        title_x=0.5,
    )


    # fig.show()
    fig_path = os.path.join(fig_dir, f'{protein_name}_{ss_id}.html')
    fig.write_html(fig_path)
    logger.info(f'Saved figure for {protein_name}_{ss_id} to dir: {fig_dir}')

def plot_vectors(vector1, vector2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(0, 0, 0, color="black", s=50)  # 原点

    ax.quiver(0, 0, 0, *vector1, color='r', label="Vector 1", linewidth=2)
    ax.quiver(0, 0, 0, *vector2, color='b', label="Vector 2", linewidth=2)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    plt.show()

def calc_vector(points):
    """
    calculate the axis of each secondary structure

    :param points:
    :return:
    orientation vector
    """
    pca = PCA(n_components=1)  # 主成分分析，n_components=1表示拟合一条直线
    pca.fit(points)

    line_direction = pca.components_[0]

    return line_direction

def check_orientation(vector1, vector2):
    """
    Check if two vectors are in the same direction.

    :param vector1: First vector
    :param vector2: Second vector
    :return: bool - True if vectors are in the same direction, False otherwise
    """
    dot_product = np.dot(vector1, vector2)
    if dot_product > 0:
        return True
    else:
        return False

def check_sheet_orientation(sheet1, sheet2, structure):
    ca_coords_sheet1 = get_ca_coords(sheet1, structure)
    vector_sheet1 = calc_vector(ca_coords_sheet1)  # first what is , then of whom

    ca_coords_sheet2 = get_ca_coords(sheet2, structure)
    vector_sheet2 = calc_vector(ca_coords_sheet2)
    return check_orientation(vector_sheet1, vector_sheet2)

def centroid(points):
    """
    Compute the centroid (midpoint) of multiple points in 3D space.

    :param points: List of tuples [(x1, y1, z1), (x2, y2, z2), ...]
    :return: A tuple representing the centroid (cx, cy, cz)
    """
    return tuple(np.mean(points, axis=0))

def distance_3d(point1, point2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    :param point1: A tuple (x1, y1, z1) representing the first point
    :param point2: A tuple (x2, y2, z2) representing the second point
    :return: Euclidean distance between the two points
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_ca_coords(sheet, structure):
    """
    Get the coordinates of the Cα atoms in a β-sheet.

    :param sheet: List of residues in the sheet
    :param structure: Structure object containing atomic coordinates
    :return: List of Cα atom coordinates
    """
    return [
        structure[(structure.res_id == res[0]) & (structure.atom_name == "CA")].coord[0]
        for res in sheet
    ]

def get_mid_point(sheet, structure):
    """
    Calculate the midpoint of a β-sheet.

    :param sheet: List of residues in the sheet
    :param structure: Structure object containing atomic coordinates
    :return: Midpoint coordinates of the sheet
    """
    return centroid(get_ca_coords(sheet, structure))

def calc_sheet_dist(sheet1, sheet2, structure):
    """
    Calculate the distance between the midpoints of two β-sheets.

    :param sheet1: List of residues in the first sheet
    :param sheet2: List of residues in the second sheet
    :param structure: Structure object containing atomic coordinates
    :return: Distance between the midpoints of the two sheets
    """

    mid_point1 = get_mid_point(sheet1, structure)
    mid_point2 = get_mid_point(sheet2, structure)
    return distance_3d(mid_point1, mid_point2)


def check_sheet_hbond(sheet1, sheet2):
    """
    Check if there is at least one hydrogen bonds between two β-sheets.

    :param sheet1: List of residues in the first sheet
    :param sheet2: List of residues in the second sheet
    :return: bool - True if hydrogen bonds exist between sheets, False otherwise
    """
    for i in sheet1:
        for j in sheet2:
            if i[0] in j[3]:
                return True
    return False


def draw_graph(G):
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=800, edge_color='gray', font_size=12,
            font_color='black')

    plt.show()

def draw_digraph(G, protein_name):
    # create pyvis direction network
    net = Network(notebook=True, directed=True, cdn_resources='in_line')

    # add nodes and edges
    for node in G.nodes():
        color = 'blue'  # 默认颜色
        if 'sheet' in node:
            color = '#ffd449'
        elif 'helix' in node:
            color = '#d62839'
        elif 'turn' in node:
            color = '#4c956c'
        # set node name
        net.add_node(node, color=color, title=node, font=dict(size=20))  # 设置字号为20

    # add edges
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])

    # layout
    net.force_atlas_2based()

    net.show(protein_name)


def create_sheet_graph(sheet_dict):
    """
    Constructs a graph representation of sheets based on their spatial relationships.

    Nodes represent individual sheets, while edges indicate either:
    - The minimum meeting distance between sheets, or
    - The presence of a hydrogen bond.

    :param sheet_dict: Dictionary containing sheet data.
    :param structure: Structural information used to determine connectivity.
    :return: A graph representation of the sheet network.
    """
    graph = nx.Graph()
    sheet_ids = sheet_dict.keys()
    for sheet_id in sheet_ids:
        graph.add_node(sheet_id)

    sheet_combinations = list(itertools.combinations(sheet_ids, 2))
    for sheet_pair in sheet_combinations:
        sheet1, sheet2 = sheet_dict[sheet_pair[0]], sheet_dict[sheet_pair[1]]

        hbond_bool = check_sheet_hbond(sheet1, sheet2)
        if hbond_bool:
            graph.add_edge(sheet_pair[0], sheet_pair[1])
    draw_graph(graph)
    input()
    return graph


def is_cycle(G, min_size=8):
    """
    Check if a graph contains a cycle with at least a minimum number (i.e. 8) of nodes.
    :param G:
    :param min_size:
    :return:
    """
    try:
        for cycle in nx.simple_cycles(G):
            if len(cycle) == min_size:
                return True
    except nx.NetworkXNoCycle:
        return False  # 没有找到环
    return False

def detect_beta_barrel(sheet_dict):
    """
    check graph whether meet the nodes >= 8
    Criteria:

    Save each β-sheet as a node, with an edge between nodes if the distance between them is less than 5 Å or if a hydrogen bond exists between them.

    A complete TIM barrel is defined as having 8 nodes.

    :return: bool: True if the structure is complete, False incomplete.
    """
    graph = create_sheet_graph(sheet_dict)
    sheet_bool = is_cycle(graph)


    return sheet_bool


def order_ss_id(ss_dict):
    """
    Order the secondary structure elements in the protein sequence.

    :param ss_dict: Dictionary containing secondary structure information.
    :return: Ordered list of secondary structure elements id.
    """
    merge_dict = {ss_id: list(map(lambda x: x[0], ss)) for i in ss_dict.values() for ss_id, ss in i.items() if ss_id }
    sorted_structures = sorted(merge_dict.items(), key=lambda x: min(x[1]))
    # sorted_structures output:
    # sheet_1: [7, 8]
    # turn_1: [10, 11, 12]

    sorted_ss = [ss[0] for ss in sorted_structures]
    return sorted_ss


def create_ss_digraph(sorted_ss):
    """
    Create a directed graph of secondary structure elements.

    :param sorted_ss: Ordered list of secondary structure elements id.
    :return: Directed graph of secondary structure elements.
    """
    digraph = nx.DiGraph()
    digraph.add_nodes_from(sorted_ss)
    for i in range(len(sorted_ss) - 1):
        digraph.add_edge(sorted_ss[i], sorted_ss[i + 1])
    return digraph


def search_digraph_motif(graph):
    """
    calculate the number of times the motif occurs in the digraph.
    """

    valid_transitions = 0
    ss_ids = list(graph.nodes)

    sheet_indices = [i for i, x in enumerate(graph) if x.startswith('sheet')]


    for i in range(len(sheet_indices) - 1):
        start, end = sheet_indices[i], sheet_indices[i + 1]
        nodes_list = list(graph.nodes)[start + 1:end]
        has_helix = any(x.startswith('helix') for x in nodes_list)
        # has_turn = any(x.startswith('turn') for x in nodes_list)
        if has_helix:
            valid_transitions += 1
        if valid_transitions == 7:
            for ss_id in ss_ids[end:]:
                if ss_id.startswith('helix'):
                    return True

    return False





def detect_alpha_barrel(ss_dict, protein_name):
    """
    Using digraph to store secondary structure.
    Meet the β-T-α-T-β motif, where there is an α-helix between two β-sheets and two Turns.
    Eventually, the structure consists of 8 α-helices and 8 β-sheets.

    :return: bool
    """
    sorted_ss = order_ss_id(ss_dict)
    digraph = create_ss_digraph(sorted_ss)
    draw_digraph(digraph, protein_name)
    barrel_bool = search_digraph_motif(digraph)
    return barrel_bool



def run(ss_dict, protein_name):
    """
    Run for filtering if the structure is a TIM barrel.
    :param protein_name:
    :param ss_dict:
    :param structure_dir:
    :return: bool
    """

    protein_name = protein_name + '.html'
    # check beta-sheet barrel
    sheet_dict = ss_dict['sheet']
    sheet_bool = detect_beta_barrel(sheet_dict)
    # check alpha-helix barrel
    alpha_bool = detect_alpha_barrel(ss_dict, protein_name)
    if sheet_bool and alpha_bool:
        return True
    else:
        return False

