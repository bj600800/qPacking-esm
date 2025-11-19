"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/15

# Description: Test fine-tuned ESM-2 model for predicting hydrophobic features
# ------------------------------------------------------------------------------
"""
import os
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import biotite.structure as struc
from biotite.structure.info import vdw_radius_single
import biotite.structure.io as strucio
from biotite.sequence.seqtypes import ProteinSequence as ps

import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, auc
test

def get_seq(atom_array):
    unique_residues = list(dict.fromkeys(atom_array.res_id))
    residue_names = [atom_array[atom_array.res_id == res_id].res_name[0] for res_id in unique_residues]
    sequence = ''.join([ps.convert_letter_3to1(res) for res in residue_names])
    return sequence

def get_hydrophobic_mask(sequence):
    hydrophobic_residues = {'A', 'V', 'I', 'L', 'M'}
    mask = [1 if aa in hydrophobic_residues else 0 for aa in sequence]
    return mask

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
    hydrophobic_mask = np.isin(structure.res_name, ["ILE", "LEU", "VAL", "ALA", "MET"]) & \
                       np.isin(structure.atom_name, ["CB", "CG1", "CG2", "CD1", "CD2", "CG", "CE"])

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

def get_seqs_labels(pdb_path_list):
    """
    Operate on hydrophobic clusters to calculate their areas and print PyMol selection commands.

    Parameters:
    structure (AtomArray): The structure to analyze.

    Returns:
    float: Total area of hydrophobic clusters.
    """
    sequences = []
    labels = []
    hydrophobic_masks = []
    for structure_file in tqdm(pdb_path_list, desc='Extracting hydrophobic labels'):
        structure = strucio.load_structure(structure_file)
        # get sequences
        sequence = get_seq(structure)
        sequences.append(sequence)
        h_mask = get_hydrophobic_mask(sequence)
        hydrophobic_masks.append(h_mask)
        # get_labels
        init_labels = [0] * len(sequence)
        first_res_id = structure.res_id[0]
        res_pairs = detect_hydrophobic_pair(structure)
        G = create_hydrophobic_graph(res_pairs)
        connected_components = nx.connected_components(G)
        for component in connected_components:
            if len(component) >= 3:
                component_subgraph = G.subgraph(component)
                node_labels = nx.get_node_attributes(component_subgraph, 'res_name')
                for res_id, _ in node_labels.items():
                    hydrophobic_pos_id = int(res_id) - int(first_res_id)
                    init_labels[hydrophobic_pos_id] = 1
        labels.append(init_labels)

    return sequences, labels, hydrophobic_masks




class HydrophobicBinaryClassificationModel(nn.Module):
    def __init__(self, backbone, num_class=2):
        super().__init__()
        self.model = backbone
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_class)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(hidden_states)
        return logits


def load_model(model_src, finetune_model_dir):
    tokenizer = EsmTokenizer.from_pretrained(finetune_model_dir)
    peft_config = PeftConfig.from_pretrained(finetune_model_dir)

    if model_src == 'esm':
        esm_model_dir = peft_config.base_model_name_or_path
        model = EsmModel.from_pretrained(esm_model_dir, add_pooling_layer=False)
    elif model_src == 'finetune':
        base_model = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
        model = PeftModel.from_pretrained(base_model, finetune_model_dir)
    else:
        raise ValueError(f"Unsupported model source: {model_src}")

    model = HydrophobicBinaryClassificationModel(backbone=model, num_class=2)
    model.classifier.load_state_dict(
        torch.load(f"{finetune_model_dir}/classifier_head.pt", weights_only=True, map_location=torch.device('mps'))
    )
    model.eval()
    return model, tokenizer


def reference(model, tokenizer, sequences):
    encoded = tokenizer(sequences, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)  #(batch_size, seq_len)
        predictions = predictions[:, 1:-1]

        return attention_mask, predictions


def eval_metrics(attention_mask, predictions, labels, hydrophobic_mask):
    valid_labels = []
    valid_preds = []
    valid_hydro = []

    # 先筛选有效位置，提取对应标签、预测和疏水信息
    for l_row, p_row, m_row, h_row in zip(labels, predictions, attention_mask, hydrophobic_mask):
        for l, p, m, h in zip(l_row, p_row, m_row, h_row):
            if m == 1:  # 只保留有效位置
                valid_labels.append(l)
                valid_preds.append(p)
                valid_hydro.append(h)

    acc = accuracy_score(valid_labels, valid_preds)  # 整体准确性

    tn, fp, fn, tp = confusion_matrix(valid_labels, valid_preds).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_negative_rate = 1 - recall

    # 细化假阳性来源统计
    fp_hydro_noncore = 0  # 假阳性中属于疏水非核心（真实负，疏水）
    fp_nonhydro = 0       # 假阳性中非疏水（真实负，非疏水）

    for l, p, h in zip(valid_labels, valid_preds, valid_hydro):
        if p == 1 and l == 0:  # 假阳性
            if h == 1:
                fp_hydro_noncore += 1
            else:
                fp_nonhydro += 1

    denom = tp + fp if (tp + fp) > 0 else 1  # 防止除零
    fp_hydro_noncore_ratio = fp_hydro_noncore / denom
    fp_nonhydro_ratio = fp_nonhydro / denom

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"FNR: {false_negative_rate}")
    print(f"FP-NonCluster: {fp_hydro_noncore_ratio}")
    print(f"FP-NonHydrophobic: {fp_nonhydro_ratio}")

    precision_curve, recall_curve, _ = precision_recall_curve(valid_labels, valid_preds)
    pr_auc = auc(recall_curve, precision_curve)
    print(f"PR AUC: {pr_auc}")


def main(model_src, finetune_model_dir, pdb_dir):
    pdb_files = list(Path(pdb_dir).glob("*.pdb"))
    seqs, labels, hydrophobic_mask = get_seqs_labels(pdb_files)
    model, tokenizer = load_model(model_src, finetune_model_dir=finetune_model_dir)
    attention_mask, predictions = reference(model, tokenizer, seqs)
    eval_metrics(attention_mask, predictions, labels, hydrophobic_mask)


if __name__ == '__main__':
    model_src = "finetune"
    finetune_model_dir = "/Users/douzhixin/Developer/qPacking/code/checkpoints/80/20250710_hydrophobic-binary_esm2-150_80_v1/best"
    pdb_dir = "/Users/douzhixin/Developer/qPacking/data/for_explaination/720_test"
    main(model_src, finetune_model_dir, pdb_dir)

