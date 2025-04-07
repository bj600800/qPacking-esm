"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/20

# Description:
# This code models the structural arrangement of a TIM-barrel domain
# where 8 beta-sheets form a central circular core, and 8 alpha-helices
# are positioned in an outer circular ring around the beta-sheets.
#
# Key Features:
# 1. Beta-sheet core: 8 beta-sheets forming a circular structure.
# 2. Helix outer ring: 8 alpha-helices surrounding the beta-sheet core.
# 3. Spatial arrangement: Helices are positioned radially outside the beta-sheets.
#
# Usage:
# - This code mainly get the secondary structures from pdb structures
# - filter.py detect the topology and connectivity of the structures.

# ------------------------------------------------------------------------------
"""
import sys
from pathlib import Path
module_path = Path('/Users/douzhixin/Developer/qPacking/code').resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

import os
import shutil
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

from qpacking.data_prepare import filter
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

#### ARGUMENTS PARSER ####
parser = argparse.ArgumentParser(description='Filter complete TIM barrel')
parser.add_argument('--dir', required=True, help='working dir')
parser.add_argument('--src', required=True, help='source dir')
args = parser.parse_args()

#### END OF ARGUMENTS PARSER ####

def get_dssp_dat(input_file_path: str, dssp_bin: str) -> object:
    # REFERENCE: https://pdb-redo.eu/dssp/about

    ss_dict = {
        'E': 'E',  # beta-sheet
        'B': 'E',  # beta-bridge
        'H': 'H',  # alpha-helix
        'G': 'H',  # 3_10 Helix
        'I': 'H',  # pi-helix
        'P': 'H',  # kapa-helix
        'T': 'T',  # Turn
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

            hbond_list = []

            # H-bonds
            N_O_1_relative_idx = int(line[38:45])
            O_N_1_relative_idx = int(line[50:56])

            for i in [N_O_1_relative_idx, O_N_1_relative_idx]:
                if i:
                    hbond_list.append(i+res_idx)

            # ss info
            ss = ss_dict[line[16]]
            dssp_dat.append([res_idx, res_name, ss, hbond_list])

        # modify the 'L' to 'H' or 'E' if the previous and next are the same.
        for i in range(1, len(dssp_dat) - 1):
            prev_ss = dssp_dat[i - 1][2]  # 前一个二级结构
            curr_ss = dssp_dat[i][2]  # 当前二级结构
            next_ss = dssp_dat[i + 1][2]  # 后一个二级结构

            if curr_ss == 'L' and prev_ss in ['H', 'E'] and next_ss in ['H', 'E']:
                if prev_ss == next_ss:  # 前后结构一致
                    dssp_dat[i][2] = prev_ss  # 将当前结构改为前后结构相同的那个
        return dssp_dat

    if stderr:
        logger.error(f"Error for {input_file_path}: {stderr}")

def get_ss_dict(dssp_dat, min_helix_aa=4, min_sheet_aa=2, min_loop_aa=2, min_turn_aa=1):
    """
    detect topology.
    :param dssp_dat:
    :param min_helix_aa:
    :param min_sheet_aa:
    :param min_loop_aa:
    :return:
    """
    ss_dict = {'helix': {}, 'sheet': {}, 'turn':{}}

    # initialize counter
    helix_counter = 1
    sheet_counter = 1
    loop_counter = 1
    turn_counter = 1

    # init current ss
    current_helix = []
    current_sheet = []
    current_loop = []
    current_turn = []

    # classify ss with dicts
    for entry in dssp_dat:
        ss_type= entry[2]
        if ss_type == 'H':
            if not current_helix:
                current_helix.append(entry)
            else:
                current_helix.append(entry)
        elif ss_type == 'E':
            if not current_sheet:
                current_sheet.append(entry)
            else:
                current_sheet.append(entry)
        # elif ss_type == 'L':
        #     if not current_loop:
        #         current_loop.append(entry)  # 开始一个新的loop
        #     else:
        #         # 如果当前氨基酸是'Loop'，且与上一个氨基酸连续，则属于同一个loop
        #         current_loop.append(entry)
        elif ss_type == 'T':
            if not current_turn:
                current_turn.append(entry)
            else:
                current_turn.append(entry)

        # if different ss, save the current chain.
        if ss_type != 'H' and current_helix:
            if len(current_helix) >= min_helix_aa:
                ss_dict['helix'][f'helix_{helix_counter}'] = current_helix
                helix_counter += 1
            current_helix = []

        if ss_type != 'E' and current_sheet:
            if len(current_sheet) >= min_sheet_aa:
                ss_dict['sheet'][f'sheet_{sheet_counter}'] = current_sheet
                sheet_counter += 1
            current_sheet = []

        # if ss_type != 'L' and current_loop:
        #     ## continues 2 loop aa
        #     if len(current_loop) >= min_loop_aa:
        #         ss_dict['loop'][f'loop_{loop_counter}'] = current_loop
        #         loop_counter += 1
        #     current_loop = []

        if ss_type != 'T' and current_turn:
            if len(current_turn) >= min_turn_aa:
                ss_dict['turn'][f'turn_{turn_counter}'] = current_turn
                turn_counter += 1
            current_turn = []

    # save the remains
    if current_helix:
        ss_dict['helix'][f'helix_{helix_counter}'] = current_helix
    if current_sheet:
        ss_dict['sheet'][f'sheet_{sheet_counter}'] = current_sheet
    # if current_loop:
    #     ss_dict['loop'][f'loop_{loop_counter}'] = current_loop
    if current_turn:
        ss_dict['turn'][f'turn_{turn_counter}'] = current_turn
    return ss_dict

def run(dssp_bin="mkdssp"):
    working_dir = args.dir
    source_dir = args.src
    log_file = os.path.join(working_dir, "processed_files.log")

    true_dir = os.path.join(working_dir, 'complete')
    os.makedirs(true_dir, exist_ok=True)

    false_dir = os.path.join(working_dir, 'incomplete')
    os.makedirs(false_dir, exist_ok=True)

    # 读取已处理文件列表
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    pdb_files = list(Path(source_dir).glob("*.pdb"))

    for structure_path in tqdm(pdb_files, desc='Filtering complete TIM barrel'):
        file_name = structure_path.stem
        if file_name in processed_files:
            logger.info(f"skip logged file: {file_name}.pdb")
            continue
        else:
            dssp_dat = get_dssp_dat(structure_path, dssp_bin=dssp_bin)
            ss_dict = get_ss_dict(dssp_dat)
            fold_bool = filter.run(ss_dict)

            if fold_bool:
                target_path = os.path.join(true_dir, file_name + '.pdb')
            else:
                target_path = os.path.join(false_dir, file_name + '.pdb')

            shutil.copy(structure_path, target_path)

            # log the processed file
            with open(log_file, "a") as f:
                f.write(file_name + "\n")

run()