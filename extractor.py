"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/7/7

# Description: 
# ------------------------------------------------------------------------------
"""
from qpacking.hydrocluster.cluster_analyzer import Analyzer

def main(pdb_dir, output_pkl_file, dssp):
    Analyzer.batch_process_pdb_files(pdb_dir, output_pkl_file, dssp)


if __name__ == '__main__':
    pdb_dir = r"/Users/douzhixin/Developer/qPacking/data/test/structure"
    output_pkl_file = r"/Users/douzhixin/Developer/qPacking/data/test/results.pkl"
    dssp = "mkdssp"
    main(pdb_dir, output_pkl_file, dssp)

