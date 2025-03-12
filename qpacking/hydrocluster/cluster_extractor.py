"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2023/07/25

# Description: hydrophobic cluster detection.
# ------------------------------------------------------------------------------
"""
import numpy as np
from collections import Counter
import networkx as nx
import biotite.structure as struc
from biotite.structure.info import vdw_radius_single
import biotite.structure.io as strucio
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def detect_hydrophobic_pair(structure, bias=1.1):
    """
    Detect pairs of hydrophobic residues within a given distance.

    Parameters:
    structure (AtomArray): The structure to analyze.
    bias (float): Distance bias added to van der Waals radii for detection.

    Returns:
    list: Sorted list of hydrophobic residue pairs.
    """
    r_vdw = vdw_radius_single("C")
    hydropho_dist = r_vdw * 2 + bias  # hydropho_dist == 4.5
    hydrophobic_mask = np.isin(structure.res_name, ["ILE", "LEU", "VAL"]) & \
                       np.isin(structure.atom_name, ["CB", "CG1", "CG2", "CD1", "CD2"])
    
    cell_list = struc.CellList(
        structure,
        cell_size=hydropho_dist,
        selection=hydrophobic_mask
        )
    
    res_pairs = set()
    for atom_idx in np.where(hydrophobic_mask)[0]:
        target_res_id = structure[atom_idx].res_id
        target_res_name = structure[atom_idx].res_name
        atoms_in_cellist = cell_list.get_atoms(coord=structure.coord[atom_idx], radius=hydropho_dist)
        potential_bond_partner_indices = [idx for idx in atoms_in_cellist if structure[idx].res_id != target_res_id]
        for potential_atom_idx in potential_bond_partner_indices:
            potential_res_id = structure[potential_atom_idx].res_id
            potential_res_name = structure[potential_atom_idx].res_name
            res_pairs.add(
                tuple(
                    sorted(
                        [(target_res_id, target_res_name),
                         (potential_res_id, potential_res_name)]
                        )
                    )
                )
    return list(sorted(res_pairs))


def create_hydrophobic_graph(res_pairs):
    """
    Create a graph representing hydrophobic interactions between residues.

    Parameters:
    res_pairs (list): List of hydrophobic residue pairs.

    Returns:
    networkx.Graph: Graph representing hydrophobic interactions.
    """
    graph = nx.Graph()
    res_list = sorted([i for res_pair in res_pairs for i in res_pair])
    
    for res in res_list:
        graph.add_node(res[0], res_name=res[1])
    
    for res1, res2 in res_pairs:
        res1_id, res2_id = res1[0], res2[0]
        graph.add_edge(res1_id, res2_id)
    
    return graph


def print_cluster_residue(node_ids):
    """
    Print the residue IDs for a cluster in a format suitable for PyMol selection.

    Parameters:
    node_ids (list): List of residue IDs in the cluster.
    """
    res_str = 'sele res ' + '+'.join([str(i) for i in node_ids])
    print(res_str)


def get_seq_len(structure):
    """
    Get the sequence length of the structure.

    Parameters:
    structure (AtomArray): The structure to analyze.

    Returns:
    int: Length of the sequence.
    """
    res_id, res_name = struc.get_residues(structure)
    seq_len = len(res_id)
    return seq_len


def run(structure_file):
    """
    Operate on hydrophobic clusters to calculate their areas and print PyMol selection commands.

    Parameters:
    structure (AtomArray): The structure to analyze.

    Returns:
    float: Total area of hydrophobic clusters.
    """
    structure = strucio.load_structure(structure_file)

    res_pairs = detect_hydrophobic_pair(structure)
    G = create_hydrophobic_graph(res_pairs)
    connected_components = nx.connected_components(G)
    
    connected_graphs = []
    
    for component in connected_components:
        if len(component) >= 3:
            component_subgraph = G.subgraph(component)
            # print(component_subgraph.nodes(data=True))
            connected_graphs.append(component_subgraph)
    return connected_graphs


if __name__ == '__main__':
    structure_file = r"/Users/douzhixin/Developer/qPacking/code/data/processed/complete/AF-A0A009ER02-F1-model_v4_TED01.pdb"
    ret = run(structure_file)