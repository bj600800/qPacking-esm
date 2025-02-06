"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/20

# Description: Identify each ss, return python API
# ------------------------------------------------------------------------------
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
last_dir_path = os.path.dirname(current_dir)
root_dir_path = os.path.dirname(last_dir_path)
sys.path.append(root_dir_path)

import subprocess
from pathlib import Path
from tqdm import tqdm

from qprotein2.spatial_geometry import quantify_ss
from qprotein2.utils import logger

logger = logger.setup_log(name=__name__)


def get_dssp_dat(input_file_path, dssp_bin):
    # REFERENCE: https://pdb-redo.eu/dssp/about

    ss_dict = {
        'E': 'E',  # beta-strand
        'B': 'E',  # beta-bridge
        'H': 'H',  # alpha-helix
        'G': 'H',  # 3_10 Helix
        'I': 'H',  # pi-helix
        'P': 'H',  # kapa-helix
        'T': 'L',  # Turn
        'S': 'L',  # Bend
        ' ': 'L',  # unstructured loop
    }

    command_args = [dssp_bin, input_file_path]

    process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout, stderr = process.communicate()

    if stdout:
        dssp_dat = []
        for line in stdout.split('\n')[28:-1]:
            # Basic identities
            res_idx = int(line[5:10])
            res_name = line[13]
            # ss info
            ss = ss_dict[line[16]]
            dssp_dat.append([res_idx, res_name, ss])

        # MODIFY DSSP_DAT with regard to the pre and last ss
        # 遍历二级结构列表，跳过第一个和最后一个元素
        for i in range(1, len(dssp_dat) - 1):
            prev_ss = dssp_dat[i - 1][2]  # 前一个二级结构
            curr_ss = dssp_dat[i][2]  # 当前二级结构
            next_ss = dssp_dat[i + 1][2]  # 后一个二级结构

            # 如果当前二级结构是'L'，并且前后都是H或E且相同
            if curr_ss == 'L' and prev_ss in ['H', 'E'] and next_ss in ['H', 'E']:
                if prev_ss == next_ss:  # 前后结构一致
                    dssp_dat[i][2] = prev_ss  # 将当前结构改为前后结构相同的那个

        return dssp_dat

    if stderr:
        logger.error(f"Error for {input_file_path}: {stderr}")


def detect_ss(dssp_dat, min_helix_num=4, min_strand_num=2, min_loop_num=2):
    ss_dict = {'helix': {}, 'strand': {}, 'loop': {}}

    # 用来处理每种二级结构类型的计数器
    helix_counter = 1
    strand_counter = 1
    loop_counter = 1

    # 存储每种类型的当前连续区域
    current_helix = []
    current_strand = []
    current_loop = []
    # 遍历原始数据并分类
    for entry in dssp_dat:
        position, amino_acid, ss_type = entry
        if ss_type == 'H':
            if not current_helix:
                current_helix.append(entry)  # 开始一个新的alpha-helix
            else:
                # 如果当前氨基酸是'Helix'，且与上一个氨基酸连续，则属于同一个alpha-helix
                current_helix.append(entry)
        elif ss_type == 'E':
            if not current_strand:
                current_strand.append(entry)  # 开始一个新的strand
            else:
                # 如果当前氨基酸是'Strand'，且与上一个氨基酸连续，则属于同一个strand
                current_strand.append(entry)
        elif ss_type == 'L':
            if not current_loop:
                current_loop.append(entry)  # 开始一个新的loop
            else:
                # 如果当前氨基酸是'Loop'，且与上一个氨基酸连续，则属于同一个loop
                current_loop.append(entry)

        # if different ss, save current chain.

        if ss_type != 'H' and current_helix:
            ## continues 4 helix
            if len(current_helix) >= min_helix_num:
                ss_dict['helix'][f'helix_{helix_counter}'] = {'residues': current_helix}
                helix_counter += 1
            current_helix = []

        if ss_type != 'E' and current_strand:
            ## continues 2 strand
            if len(current_strand) >= min_strand_num:
                ss_dict['strand'][f'strand_{strand_counter}'] = {'residues': current_strand}
                strand_counter += 1
            current_strand = []

        if ss_type != 'L' and current_loop:
            ## continues 2 loop
            if len(current_loop) >= min_loop_num:
                ss_dict['loop'][f'loop_{loop_counter}'] = {'residues': current_loop}
                loop_counter += 1
            current_loop = []

    # save the left
    if current_helix:
        ss_dict['helix'][f'helix_{helix_counter}'] = {'residues': current_helix}
    if current_strand:
        ss_dict['strand'][f'strand_{strand_counter}'] = {'residues': current_strand}
    if current_loop:
        ss_dict['loop'][f'loop_{loop_counter}'] = {'residues': current_loop}

    return ss_dict


def run(structure_dir, dssp_bin, plot_fig):
    pdb_files = list(Path(structure_dir).glob("*.pdb"))
    output = {}
    for structure_path in tqdm(pdb_files, desc='computing secondary structures'):
        file_name = structure_path.stem
        dssp_dat = get_dssp_dat(structure_path, dssp_bin=dssp_bin)
        ss_dict = detect_ss(dssp_dat)
        ss_data = quantify_ss.run(file_name, ss_dict, structure_dir, plot_fig)
        output[file_name] = ss_data
    return output
