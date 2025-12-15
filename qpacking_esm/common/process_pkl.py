"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/18

# Description: protein feature matrix
# ------------------------------------------------------------------------------
"""
import os
from tqdm import tqdm
import pickle
from qpacking_esm.common import logger

logger = logger.setup_log(name=__name__)

residue_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def load_existing_results(pkl_file):
    """
    Load existing results from a pickle file.
    :param pkl_file: existing results file
    :return: a dictionary containing loaded results
    """
    try:
        with open(pkl_file, "rb") as f:
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


def replace_pkl(all_pkl, addit_pkl, output_pkl):
    all_features = load_existing_results(all_pkl)
    addit_features = load_existing_results(addit_pkl)

    to_delete = []

    for protein_name, feature in tqdm(all_features.items()):
        if protein_name in addit_features:
            all_features[protein_name]['rsa'] = addit_features[protein_name]['rsa']
        else:
            to_delete.append(protein_name)

    for protein_name in to_delete:
        del all_features[protein_name]

    with open(output_pkl, "wb") as f:
        pickle.dump(all_features, f)

def merge_pkl(pkl1, pkl2):
    """
    dict_keys(['class', 'rsa', 'bsa', 'degree', 'order', 'resid_name', 'length'])
    Args:
        pkl1:
        pkl2:

    Returns:

    """
    feature1 = load_existing_results(pkl1)
    feature2 = load_existing_results(pkl2)

    output_pkl = {**feature1, **feature2}
    return output_pkl

def split_feature(feature, key, data_type):
    single_feature = {}
    for k, v in feature.items():
        length = v['length']
        if key=='class':
            single_feature[k] = v[key]

        elif key == 'bsa':
            single_feature[k] = {key: value/length for key, value in v[key].items()}

        elif key=='order':
            single_feature[k] = {key: value/length for key, value in v[key].items()}

        elif key=='resid_name':
            seq_dict = {key: residue_3to1[value] for key, value in v[key].items()}
            sequence = ''.join([seq_dict[k] for k in sorted(seq_dict.keys())])
            single_feature[k] = {'seq': sequence, 'seq_dict': seq_dict}
        else:
            if data_type == 'float32':
                single_feature[k] = {key: float(value) for key, value in v[key].items()}
            elif data_type == 'int':
                single_feature[k] = {key: value for key, value in v[key].items()}
    return single_feature

def run_split(input_pkl):
    dir_path = os.path.dirname(input_pkl)
    file_name = os.path.basename(input_pkl)
    existing_results = load_existing_results(input_pkl)
    feature_names = {'class': 'int', 'bsa': 'float32', 'degree': 'int', 'rsa': 'float32', 'order': 'float32', 'resid_name': 'str'}
    for name, data_type in tqdm(feature_names.items()):
        try:
            new_pkl = os.path.join(dir_path, file_name.split('.')[0]+f'_{name}.pkl')
            new_feature = split_feature(existing_results, name, data_type)
            with open(new_pkl, "wb") as f:
                pickle.dump(new_feature, f)
        except:
            print(name)

def get_example_data(input_pkl):
    dir_path = os.path.dirname(input_pkl)
    file_name = os.path.basename(input_pkl)
    existing_results = load_existing_results(input_pkl)
    feature_names = {'class': 'int', 'bsa': 'float32', 'degree': 'int', 'rsa': 'float32', 'order': 'float32',
                     'resid_name': 'str'}
    test_protein_name = []
    for name, data_type in tqdm(feature_names.items()):
        new_pkl = os.path.join(dir_path, f'test/{file_name.split(".")[0]}_{name}.pkl')
        os.makedirs(os.path.dirname(new_pkl), exist_ok=True)
        new_feature = split_feature(existing_results, name, data_type)
        if name == 'class':
            test_protein_name = list(new_feature.keys())[:10]
        save_feature = {protein_name: feature for protein_name, feature in new_feature.items() if
                        protein_name in test_protein_name}
        with open(new_pkl, "wb") as f:
            pickle.dump(save_feature, f)

def run_merge(pkl1, pkl2, output_pkl_path):
    output_pkl = merge_pkl(pkl1, pkl2)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(output_pkl, f)
    run_split(output_pkl_path)


if __name__ == '__main__':
    output_feature_pkl = r"/Users/douzhixin/Developer/qPacking-esm/data/benchmark/if1/if1.pkl"
    with open(output_feature_pkl, 'rb') as f:
        loaded_data = pickle.load(f)
    print(loaded_data)







