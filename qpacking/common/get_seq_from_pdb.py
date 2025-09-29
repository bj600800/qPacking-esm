# ------------------------------------------------------------------------------
# Author:    Dou zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/20
#
# Description: An api-tool
# ------------------------------------------------------------------------------
import os
import argparse
import time

from tqdm import tqdm
import biotite.structure.io as strucio
from biotite.sequence.seqtypes import ProteinSequence as ps

import logger

logger = logger.setup_log(name=__name__)

#### ARGUMENTS PARSER ####
parser = argparse.ArgumentParser(description='Get sequences from pdbs')

parser.add_argument('--dir', required=True, help='PDB dir')
parser.add_argument('--fasta_file', required=True, help='fasta file')

args = parser.parse_args()
#### END OF ARGUMENTS PARSER ####


def get_array(pdb_structure):
    return strucio.load_structure(pdb_structure)


def get_seq(atom_array):
    unique_residues = list(dict.fromkeys(atom_array.res_id))
    residue_names = [atom_array[atom_array.res_id == res_id].res_name[0] for res_id in unique_residues]
    sequence = ''.join([ps.convert_letter_3to1(res) for res in residue_names])
    return sequence


def main(pdb_dir, fasta_file):
    dir_path = os.path.dirname(fasta_file)
    os.makedirs(dir_path, exist_ok=True)

    time1 = time.time()
    seq_dict = {}

    logger.info('Reading sequences from pdb files')
    for pdb_file in tqdm(os.listdir(pdb_dir), desc='Reading sequences'):
        header = pdb_file.rstrip('.pdb').rstrip('.cif')
        pdb_structure = os.path.join(pdb_dir, pdb_file)
        try:
            atom_array = get_array(pdb_structure)
            sequence = get_seq(atom_array)
            seq_dict[header] = sequence
        except:
            pass

    logger.info(f'Writing sequences to {fasta_file}')
    with open(fasta_file, 'w') as f:
        for header, seq in seq_dict.items():
            f.write(f'>{header}\n{seq}\n')

    time2 = time.time()
    logger.info(f'Writing complete! \nUsing {int(time2-time1)} seconds.')


if __name__ == '__main__':
    pdb_structure = args.dir
    fasta_file = args.fasta_file
    main(pdb_structure, fasta_file)

