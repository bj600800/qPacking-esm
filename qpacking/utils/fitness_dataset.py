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
    seq = ''
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
    return seq


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

def get_dataset(seq, data_list, offset_idx):
    dataset = []
    for data in data_list:
        original_pos = data[1]
        pos = int(original_pos) - offset_idx
        seq_id = data[0]+ '_' + data[2]+str(pos+1)+data[3]
        wt_seq = seq
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
    csv_path = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/ss.csv"
    fasta_path = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/ss.fasta"
    pkl_path = r"/Users/douzhixin/Developer/qPacking/data/benchmark/tim-db/ss.pkl"
    offset_idx = 44
    data_list = read_csv(csv_path)
    seq = read_seq(fasta_path)
    dataset = get_dataset(seq, data_list, offset_idx)
    dump_pkl(dataset, pkl_path)

    with open(pkl_path, 'rb') as f:
        loaded_data = pickle.load(f)

    print(loaded_data[0])
