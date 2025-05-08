"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/5/1

# Description: 
# ------------------------------------------------------------------------------
"""
import pickle

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
        print(e)
        return {}

if __name__ == '__main__':
    output_pkl_file = r"/Users/douzhixin/Developer/qPacking/data/70_results.pkl"
    load_existing_results = load_existing_results(output_pkl_file)
    print(len(load_existing_results))
    # print(load_existing_results)
    for k, v in list(load_existing_results.items())[:2]:
        print(k, v)