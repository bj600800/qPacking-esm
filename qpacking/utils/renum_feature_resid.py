"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/6

# Description: 
# ------------------------------------------------------------------------------
"""
import pickle
from pathlib import Path

from tqdm import tqdm

import biotite.structure.io as strucio
import biotite.structure as struc
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

def load_existing_results(output_file):
    """
    Load existing results from a pickle file.
    :param output_file: existing results file
    :return: a dictionary containing loaded results
    """
    try:
        with open(output_file, "rb") as f:
            results_dict = pickle.load(f)  # output file only 1 obj.
            if not isinstance(results_dict, dict):
                return {}
            return results_dict
    except (FileNotFoundError, EOFError):
        return {}
    except Exception as e:
        logger.error(e)
        return {}

def get_first_residue_id(structure):
    """
    Get the first residue ID from the structure.

    Parameters:
    structure (AtomArray): The structure to analyze.

    Returns:
    str: The ID of the first residue.
    """
    return structure.res_id[0]

def renumber_resid(feature, first_res_id):
    """
    Renumber the residue IDs in the structure feature.

    Parameters:
    feature (dict): The structure feature to renumber.

    Returns:
    dict: The renumbered structure feature.
    """
    new_feature = {}
    for res_id, feat in feature.items():
        if not isinstance(res_id, str):
            res_id = int(res_id) - int(first_res_id)
            new_feature[res_id] = feat
        else:
            continue
    return new_feature


def save_to_pickle(data, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

def run(pdb_dir, pkl_file, new_file):
    new_feature = {}
    renumber_count = 0
    feature = load_existing_results(pkl_file)
    pdb_files = list(Path(pdb_dir).glob("*.pdb"))
    for pdb in tqdm(pdb_files):
        pdb_name = pdb.stem
        structure = strucio.load_structure(str(pdb))
        first_res_id = get_first_residue_id(structure)
        _ = {}
        if first_res_id != "1":
            _ = protein_feature = feature.get(pdb_name, None)
            if protein_feature:
                feature_class = protein_feature['class']
                renum_feature = renumber_resid(feature_class, first_res_id)
                new_feature[pdb_name] = renum_feature

                # print(pdb, first_res_id)
                # print("original: ", feature_class)
                # print("renumbered: ", renum_feature)
                renumber_count += 1
        else:
            if _:
                new_feature[pdb_name] = _
    logger.info(f"Renumbered {renumber_count} structures out of {len(pdb_files)}")
    save_to_pickle(new_feature, new_file)
    return new_feature


if __name__ == '__main__':
    pdb_dir = r"/Users/douzhixin/Developer/qPacking/data/structure"
    pkl_file = r"/Users/douzhixin/Developer/qPacking/data/results.pkl"
    new_pkl = r"/Users/douzhixin/Developer/qPacking/data/results_renum.pkl"
    run(pdb_dir, pkl_file, new_pkl)