"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/7/27

# Description: 
# ------------------------------------------------------------------------------
"""
import csv
import pickle
from Bio import SeqIO


def read_seq(fasta_path):
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        seq = str(record.seq)
        seq_dict[seq_id] = seq
    return seq_dict


def read_csv(csv_path):
    data_list = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            data_list.append(row)
    return data_list

def replace_char(s, index, new_char):
    return s[:index] + new_char + s[index+1:]

def get_dataset(seq_dict, data_list, offset_idx):
    dataset = []
    for data in data_list:
        original_pos = data[1]
        pos = int(original_pos) - offset_idx
        seq_name = data[0]
        seq_id = data[0]+ '_' + data[2]+str(pos+1)+data[3]
        wt_seq = seq_dict[seq_name]
        mt_seq = replace_char(wt_seq, pos, data[3])
        fitness = float(data[4])
        dataset.append({
            'id': seq_id,
            'wt_seq': wt_seq,
            'mt_seq': mt_seq,
            'fitness': fitness})
    return dataset

def dump_pkl(dataset, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {pkl_path}")


if __name__ == '__main__':
    import os
    csv_path = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/tm.csv"
    fasta_dir = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db"
    pkl_path = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/tm.pkl"
    offset_idx = 40
    data_list = read_csv(csv_path)
    fasta_files = [os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) if f.endswith('.fasta')]
    fasta_dict = {}
    for file in fasta_files:
        _dict = read_seq(file)
        fasta_dict.update(_dict)
    dataset = get_dataset(fasta_dict, data_list, offset_idx)
    dump_pkl(dataset, pkl_path)

    with open(pkl_path, 'rb') as f:
        loaded_data = pickle.load(f)

