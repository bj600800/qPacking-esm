"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/20

# Description: 
# ------------------------------------------------------------------------------
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx

from qpacking.util import logger

logger = logger.setup_log(name=__name__)


def get_dssp_dat(input_file_path):
    print(input_file_path)
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
    
    command_args = [dssp_path, '-i', input_file_path]
    print(command_args)
    process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout, stderr = process.communicate()
    # if process.returncode != 0:
    #     print("Error:", stderr)
    # else:
    #     print("Output:", stdout)

    if stdout:
        dssp_dat = []
        for line in stdout.split('\n')[28:-1]:
            # Basic identities
            idx = int(line[:5])
            res_idx = int(line[5:10])
            res_name = line[13]
            
            # ss info
            ss = ss_dict[line[16]]
            
            # Hbond
            N_O_1_relative_idx = int(line[38:45])
            O_N_1_relative_idx = int(line[50:56])
            N_O_2_relative_idx = int(line[61:67])
            O_N_2_relative_idx = int(line[72:78])
            
            hbond = {'NO1': N_O_1_relative_idx + res_idx,
                     'ON1': O_N_1_relative_idx + res_idx,
                     'NO2': N_O_2_relative_idx + res_idx,
                     'ON2': O_N_2_relative_idx + res_idx
                     
                     }
            
            # Alpha Carbon coords
            xc = float(line[115:122])
            yc = float(line[122:129])
            zc = float(line[129:136])
            
            dssp_dat.append([res_idx, ss, hbond, (xc, yc, zc)])
        return dssp_dat
    
    if stderr:
        logger.error(f"Error for {input_file_path}: {stderr}")


def detect_beta_barrel(dssp_dat):
    sheet_pos = []
    beta_sheet_8 = []  # beta1, beta2, beta3,..., beta8
    current_dat = dssp_dat[0]  # 初始化第一个data
    current_group = [current_dat]  # 初始化当前组
    segments = []
    for dat in dssp_dat[1:]:
        ss = dat[1]  # discriminate value
        if ss == 'E':
            sheet_pos.append(dat[0])
        if ss == current_dat[1]:
            current_group.append(dat)
        else:
            segments.append({current_dat[1]: current_group})
            current_group = [dat]
            current_dat = dat

    for seg in segments:
        for key, residue_info_list in seg.items():
            
            # num of E more than 2 continuously
            if key == 'E' and len(residue_info_list) >= 2:
                vector_list = []
                for info in residue_info_list:
                    
                    hbond = info[2]
                    coord_a = info[3]
                    parters = hbond.values()
                    
                    for p_idx in parters:
                        coord_b = dssp_dat[p_idx][3]

                        if p_idx in sheet_pos:
                            vector = calculate_vector(coord_b, coord_a)
                            vector_list.append(vector)
                print(residue_info_list[0][0], residue_info_list[-1][0], len(vector_list))
                for i in vector_list:
                    for j in vector_list:
                        if i != j:
                            if calculate_dot(i,j) < 0:
                                print(i, j)
                                beta_sheet_8.append(seg)
                                break
                    break
            break
    print(len(beta_sheet_8))
    
                            



def calculate_vector(point_B, point_A):
    return (point_B[0] - point_A[0], point_B[1] - point_A[1], point_B[2] - point_A[2])


def calculate_dot(vector1, vector2):
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    # 计算向量的点积
    dot_product = np.dot(v1, v2)
    
    return dot_product


def run(structure_path):
    dssp_dat = get_dssp_dat(structure_path)
    detect_beta_barrel(dssp_dat)


def batch_run(struct_dir):
    """
    批量处理指定目录中的结构文件，生成 DSSP 数据。

    Args:
        struct_dir (str): 包含结构文件的目录路径。
    """
    
    num_cores = os.cpu_count()
    logger.info(f"{num_cores} threads running.")
    
    input_files = [os.path.join(struct_dir, i) for i in os.listdir(struct_dir)]
    # for i in input_files:
    #     run(i)

    # for batch run
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(run, file) for file in input_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DSSP"):
            future.result()
            # try:
            #     future.result()
            # except Exception as e:
            #     logger.error(f"Error processing file: {e}")



dssp_path = "mkdssp"

struct_dir = r"/test"
batch_run(struct_dir)
