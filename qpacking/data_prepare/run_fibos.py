"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/5/5

# Description: 
# ------------------------------------------------------------------------------
"""
# conda env fibos

from concurrent.futures import ProcessPoolExecutor, as_completed
import fibos
import os
from pathlib import Path
import time
import pandas as pd

# 并行工作函数：计算 FIBOS + OSP，并返回结果
def process_pdb(pdb_path, fibos_folder, method="FIBOS"):
    start = time.time()
    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem[-4:].lower()
    srf_file = f"prot_{pdb_id}.srf"
    srf_path = os.path.join(fibos_folder, srf_file)

    # Run FIBOS
    fibos.occluded_surface(str(pdb_path), method=method)

    # Run OSP
    osp_result = fibos.osp(srf_path)

    elapsed = time.time() - start
    return pdb_path.name, elapsed, osp_result

if __name__ == "__main__":
    pdb_folder = "/Users/douzhixin/Developer/qPacking/data/structure"
    fibos_folder = "fibos_files"
    os.makedirs(fibos_folder, exist_ok=True)

    # 收集 PDB 文件路径
    pdb_paths = list(Path(pdb_folder).glob("*.pdb"))
    ideal_cores = 10 #min(os.cpu_count(), len(pdb_paths))

    with ProcessPoolExecutor(max_workers=ideal_cores) as executor:
        futures = [
            executor.submit(process_pdb, path, fibos_folder)
            for path in pdb_paths
        ]
        for future in as_completed(futures):
            try:
                name, elapsed, osp_df = future.result()
                print(f"{name} finished in {elapsed:.2f} seconds")
                print(osp_df.head(3))  # 打印前三行
                print("-" * 40)
            except Exception as e:
                print(f"Error processing file: {e}")
