"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/3/12

# Description: Calculates various Hydrophobic Packing Descriptors (HPDs) from PDB files in parallel using Producer-Consumer.
# 1. Extract hydrophobic clusters from a PDB file.
# 2. Calculate different metrics for each cluster at the residue level.
# 3. Run batch calculations for all PDB files in a directory.
# 4. Keep original data.
# 4. save raw data to a pickle file -> results.pkl
# ------------------------------------------------------------------------------
"""

import networkx as nx
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from biotite.structure import sasa

# from qpacking_esm.common import visualization
from qpacking.common.logger import setup_log

logger = setup_log(name=__name__, enable_file_log=False)


class Analyzer:
    def __init__(self, cluster_graphs, structure, pdb_file, dssp):
        self.pdb_file = pdb_file
        self.cluster_graphs = cluster_graphs
        self.structure = structure
        self.protein_res_id = [int(i) for i in set(self.structure.res_id)]
        self.protein_length = len(self.protein_res_id)
        self.first_res_id = structure.res_id[0]
        self.dssp = dssp

    @staticmethod
    def min_max_normalizer(feature_dict, min_value=0.0):
        """
        Normalize the feature_dict values to [min_value, 1.0].

        Args:
            feature_dict (dict): Dictionary of {res_id: feature_value}.
            min_value (float): The lower bound of normalization (default: 0.0).

        Returns:
            dict: {res_id: normalized_feature_value}
        """
        if not feature_dict:
            return {}

        keys = list(feature_dict.keys())
        values = np.array(list(feature_dict.values()), dtype=np.float32)

        min_val = np.min(values)
        max_val = np.max(values)

        if max_val == min_val:
            # All values are the same → assign to 1.0
            return {k: 1.0 for k in keys}

        norm_values = min_value + (1.0 - min_value) * (values - min_val) / (max_val - min_val)
        norm_feature = dict(zip(keys, norm_values))
        return norm_feature

    def get_class(self, G, i):
        node_labels = nx.get_node_attributes(G, 'res_name')
        class_feature = {}
        for res_id, _ in node_labels.items():
            res_id = int(res_id) - int(self.first_res_id)  # position id. position start from 0

            # residue id start from 0, cluster id start from 1
            class_feature[res_id] = i
        return class_feature

    def get_area(self):
        """
        calc area for each residue in the protein.
        area_i = SASA_i / protein_length
        """
        area_feature = {}
        for res_id in self.protein_res_id:
            residue_mask = np.isin(self.structure.res_id, res_id)
            res_array = self.structure[residue_mask]
            if len(res_array) == 0:
                continue
            res_id = int(res_id) - int(self.first_res_id)
            # reverse area values, higher value means more stable
            area_feature[res_id] = float(sum(sasa(res_array)) / self.protein_length)
        return area_feature

    def get_degree(self, G):
        # Normalize [0, 1] the degree of each node in the whole dataset with minimum and maximum degree.
        degree_feature = {}
        degree_dict = dict(G.degree())
        for node, degree in degree_dict.items():
            node = int(node) - int(self.first_res_id)
            degree_feature[node] = degree
        # visualization.draw_graph_interactive(G, "degree_graph.html")
        # input()
        degree_feature = {k: -np.log(v) for k, v in degree_feature.items()}
        return degree_feature

    def get_rsa(self, packing_res):
        """
        Calculate the relative solvent accessibility (RSA) for a list of residues using biopython-dssp-rasa.
        Higher value means more favorable [0.1, 1].
        :param resi_list: List of residue indices
        :return: {res_i: rsa} dictionary
        """
        p = PDBParser()
        structure = p.get_structure("protein", self.pdb_file)
        model = structure[0]
        dssp = DSSP(model, self.pdb_file, dssp=self.dssp)

        # create dssp rsa dict
        dssp_rsa_dict = {
            key[1][1]: dssp[key][3]
            for key in dssp.keys()
        }
        # get rsa
        rsa_dict = {
            int(res) - int(self.first_res_id): np.log(dssp_rsa_dict.get(res, None) + 1.0e-6)  # avoid log(0)
            for res in packing_res
        }

        return rsa_dict

    def get_order(self, G):
        """
        Normalization depend on the distribution of order values in the dataset. in dataset.py normalize
        Packing order (PO) defines the difficulties of packing for each hydrophobic residue.

        The more difficult to pack, the higher the order.
        cite: https://doi.org/10.1006/jmbi.1998.1645
        :return: {res_id: PO} dictionary
        """
        # visualization.show_hydrocluster_pymol(self.pdb_file, self.cluster_graphs)
        # input()
        hydro_res = G.nodes()
        l_protein = self.protein_length  # question: bigger num refers to more residues or more difficult to pack.
        order_dict = {}
        for i in hydro_res:
            neighbors = list(G.neighbors(i))
            degree_i = len(neighbors)  # |C_i|
            if degree_i == 0:
                continue
            dist_sum = sum(abs(int(i) - int(j)) for j in neighbors)  # sum |i-j|
            order_i = np.log((dist_sum / (degree_i * l_protein)) + 1.0e-6)  # 1 / (|C_i| * L) * sum |i-j|
            res = int(i) - int(self.first_res_id)
            order_dict[res] = order_i
        return order_dict

    # def get_centrality(self, G):
    #     """
    #     Compute the centrality of each hydrophobic residue in the cluster.
    #     Centrality is defined as the CA distance to the cluster centroid.
    #     Normalized to [0.1, 1.0] per cluster, where smaller values indicate closer to center.
    #
    #     :param G: NetworkX graph of a hydrophobic cluster
    #     :return: dict {res_id: centrality_score}
    #     """
    #     centrality_dict = {}
    #     res_list = list(G.nodes())
    #     coords = []
    #
    #     # Get CA atom coordinate for each residue
    #     for res_id in res_list:
    #         residue_mask = np.isin(self.structure.res_id, res_id) & (self.structure.atom_name == 'CA')
    #         coord = self.structure[residue_mask].coord
    #         coords.append(coord[0])
    #
    #     coords = np.array(coords)  # shape: (N, 3)
    #     centroid = np.mean(coords, axis=0)  # geometric center
    #
    #     # Compute distances to center
    #     distances = np.linalg.norm(coords - centroid, axis=1)  # shape: (N,)
    #
    #     for i, res_id in enumerate(res_list):
    #         res_idx = int(res_id) - int(self.first_res_id)  # optional: map to 0-based
    #         centrality_dict[res_idx] = float(distances[i])
    #     centrality_dict = self.min_max_normalizer(centrality_dict, min_value=0)
    #     return centrality_dict

    def run(self):
        packing_res = []
        struct_features = {'class': {},
                           'area': {},  # dataset.py normalize
                           'degree': {},
                           'rsa': {},
                           'order': {},  # dataset.py normalize
                           'centrality': {}}

        try:
            for i, G in enumerate(self.cluster_graphs):
                # i starts from 0, but cluster index starts from 1
                class_feature = self.get_class(G, i + 1)
                area_feature = self.get_area()
                degree_feature = self.get_degree(G)
                order_feature = self.get_order(G)
                centrality_feature = self.get_centrality(G)
                packing_res.extend(list(G.nodes()))
                struct_features['class'].update(class_feature)  # 1
                struct_features['area'].update(area_feature)  # 2
                struct_features['degree'].update(degree_feature)  # 3
                struct_features['order'].update(order_feature)  # 4
                struct_features['centrality'].update(centrality_feature)  # 5
            rsa_dict = self.get_rsa(packing_res)
            struct_features['rsa'] = rsa_dict  # 6

        except Exception as e:
            logger.warning(f"Error in calculating {self.pdb_file}: {e}")
            return False

        # clean cache
        self.cluster_graphs = None
        self.structure = None
        if all(v for v in struct_features.values()):
            return struct_features
        else:
            return False
