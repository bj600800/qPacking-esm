"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/4

# Description: hydrophobic amino acid buried surface area (BSA) calculation
# ------------------------------------------------------------------------------
"""
import pickle
import os
import biotite.structure.io as strucio
from scipy import stats
from tqdm import tqdm
from qpacking.common import logger
logger = logger.setup_log(name=__name__)

# MaxASA (Å²) from Tien et al., PLOS ONE 2013, doi:10.1371/journal.pone.0080635
max_sasa_dict = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 223.0,
    "GLU": 225.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0
}

def load_existing_results(output_pkl):
    """
    Load existing results from a pickle file.
    :param output_file: existing results file
    :return: a dictionary containing loaded results
    """
    try:
        with open(output_pkl, "rb") as f:
            results_dict = pickle.load(f)  # output file only 1 obj.
            if not isinstance(results_dict, dict):
                return {}
            return results_dict
    except (FileNotFoundError, EOFError):
        logger.error('FileNotFoundError')
        return {}
    except Exception as e:
        logger.error(e)
        return {}

def get_bsa(result_dict, structure_dir, pkl_file):
    output_result_dict = result_dict.copy()
    for protein_name, ret_dict in tqdm(result_dict.items()):
        pdb_path = os.path.join(structure_dir, protein_name + '.pdb')
        structure = strucio.load_structure(pdb_path)
        residue_id = [int(i) for i in set(structure.res_id)]
        unique_residues = list(dict.fromkeys(structure.res_id))
        residue_name = [str(structure[structure.res_id == res_id].res_name[0]) for res_id in unique_residues]
        residue_dict = dict(zip(residue_id, residue_name))
        rsa_dict = ret_dict['rsa']
        bsa_dict = {}
        for res_id, rsa in rsa_dict.items():
            res_name = residue_dict.get(res_id)
            max_sasa = max_sasa_dict.get(res_name)
            bsa = (1-rsa) * max_sasa
            bsa_dict[res_id] = bsa
        output_result_dict[protein_name].update({'bsa':bsa_dict})
        del output_result_dict[protein_name]['area']
    with open(pkl_file, 'wb') as f:
        pickle.dump(output_result_dict, f)
    print("new pkl file generated")
    return output_result_dict

def area_analysis(result_dict):
    high_temp_area = []
    low_temp_area = []
    for protein_name, ret_dict in result_dict.items():
        temperature = ret_dict['temperature']
        seq_lenth = ret_dict['length']
        avg_area = sum(list(ret_dict['bsa'].values()))/seq_lenth
        if temperature >= 50:
            high_temp_area.append(avg_area)
        elif temperature < 50:
            low_temp_area.append(avg_area)
    t_stat, p_value = stats.ttest_ind(high_temp_area, low_temp_area)

    print(len(high_temp_area))
    print(len(low_temp_area))
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.3f}")

    if p_value < 0.05:
        print("差异显著 (p < 0.05)")
    else:
        print("差异不显著 (p ≥ 0.05)")

    mean1 = sum(high_temp_area) / len(high_temp_area)
    mean2 = sum(low_temp_area) / len(low_temp_area)

    print(f"high_temp_area 平均值: {mean1:.2f}")
    print(f"low_temp_area 平均值: {mean2:.2f}")

    if mean1 > mean2:
        print("high_temp_area 高于 low_temp_area")
    elif mean1 < mean2:
        print("low_temp_area 高于 high_temp_area")
    else:
        print("两组平均值相等")
    return high_temp_area, low_temp_area

if __name__ == '__main__':
    pkl_file = r"/Users/douzhixin/Developer/qPacking/Data/70_feature/new/70.pkl"
    structure_dir = r"/Users/douzhixin/Developer/qPacking/Data/70_structure/complete"
    result_dict = load_existing_results(pkl_file)
    print(result_dict)
    # new_result_dict = get_bsa(result_dict, structure_dir, pkl_file)
    # area_analysis(new_result_dict)


