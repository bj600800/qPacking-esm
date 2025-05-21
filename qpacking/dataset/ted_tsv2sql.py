import sqlite3
import csv
import os
import argparse
from tqdm import tqdm

#### ARGUMENTS PARSER ####
parser = argparse.ArgumentParser(description='create db')

parser.add_argument('--tsv', required=True, help='ted tsv path')
parser.add_argument('--cath', required=True, help='cath label')
parser.add_argument('--dir', required=True, help='working dir')


args = parser.parse_args()

#### END OF ARGUMENTS PARSER ####
cath_id = args.cath
wk_dir = args.dir
# 连接到SQLite数据库（如果数据库不存在，会自动创建）
conn = sqlite3.connect(os.path.join(wk_dir, f"{cath_id}.db"))
cursor = conn.cursor()

# 创建表格
cursor.execute('''
CREATE TABLE IF NOT EXISTS summary (
    ted_id TEXT PRIMARY KEY,
    md5 TEXT,
    consensus_level TEXT,
    chopping TEXT,
    nres_domain INTEGER,
    num_segments INTEGER,
    plddt REAL,
    num_helix_strand_turn INTEGER,
    num_helix INTEGER,
    num_strand INTEGER,
    num_helix_strand INTEGER,
    num_turn INTEGER,
    proteome_id TEXT,
    cath_label TEXT,
    cath_assignment_level TEXT,
    cath_assignment_method TEXT,
    packing_density REAL,
    norm_rg REAL,
    tax_common_name TEXT,
    tax_scientific_name TEXT,
    tax_lineage TEXT 
)
''')

# 定义插入命令
insert_sql = '''
INSERT OR REPLACE INTO summary (ted_id, md5, consensus_level, chopping, nres_domain, num_segments, plddt, num_helix_strand_turn, num_helix, num_strand, num_helix_strand, num_turn, proteome_id, cath_label, cath_assignment_level, cath_assignment_method, packing_density, norm_rg, tax_common_name, tax_scientific_name, tax_lineage)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
'''

# 读取TSV文件并批量插入数据
batch_size = 500000  # 每批插入的行数
total_rows = 364806077
tsv_path = args.tsv
with open(tsv_path, 'r') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    batch = []

    for row in tqdm(tsv_reader, total=total_rows):
        if row[13] == cath_id:
            batch.append(row)
        # 当批量达到指定大小时，插入数据
        if len(batch) >= batch_size:
            cursor.executemany(insert_sql, batch)
            batch = []  # 清空批量数据

    # 插入剩余的行（如果有）
    if batch:
        cursor.executemany(insert_sql, batch)

print('creating index')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_cath_label ON summary (cath_label)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_chopping ON summary (chopping)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_nres_domain ON summary (nres_domain)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_num_segments ON summary (num_segments)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_plddt ON summary (plddt)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_consensus_level ON summary (consensus_level)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_packing_density ON summary (packing_density)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_norm_rg ON summary (norm_rg)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_nres_domain ON summary (nres_domain)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_num_helix_strand_turn ON summary (num_helix_strand_turn)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_num_helix ON summary (num_helix)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_num_strand ON summary (num_strand)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_num_helix_strand ON summary (num_helix_strand)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_num_turn ON summary (num_turn)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_tax_scientific_name ON summary (tax_scientific_name)')
conn.commit()
print('creating done')

# 提交更改并关闭连接
cursor.close()
conn.close()
