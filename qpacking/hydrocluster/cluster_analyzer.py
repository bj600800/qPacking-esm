"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/3/12

# Description: This script calculates various metrics for hydrophobic clusters from PDB files in batch mode.
# 1. Extract hydrophobic clusters from a PDB file.
# 2. Calculate different metrics for each cluster.
# 3. Run batch calculations for all PDB files in a directory.
# 4. save raw data to a pickle file -> results.pkl
# ------------------------------------------------------------------------------
"""
import os
import pickle
import concurrent.futures
import multiprocessing
import threading
import time
from tqdm import tqdm
from pathlib import Path

from qpacking.hydrocluster import cluster_extractor
from qpacking.utils import logger
logger = logger.setup_log(name=__name__)

class Analyzer:
    def __init__(self, clusters_graph, structure):
        self.clusters_graph = clusters_graph
        self.structure = structure

    def get_cluster_info(self):
        i = 0
        for i in range(1000):
            i+=1
        print(self.clusters_graph)

    def calc_contact_count(self):
        print("Calculating contact count...")
        pass

    def run(self):
        self.get_cluster_info()
        self.calc_contact_count()

    @staticmethod
    def process_pdb_file(pdb_file, queue, processed_log, log_file):
        """producer for one file"""
        if pdb_file in processed_log:
            print(f"Skipping {pdb_file}, already processed.")
            return

        clusters_graph = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
        analyzer = Analyzer(clusters_graph, pdb_file)
        result = analyzer.run()
        queue.put((pdb_file, result))  # 结果入队列

        # Task logging
        Analyzer.log_processed_file(pdb_file, log_file)  # 记录到日志文件

    @staticmethod
    def file_writer(queue, output_file):
        """consumer: write pkl file in FIFO way"""
        with open(output_file, "ab") as f:  # 追加模式
            while True:
                result = queue.get()
                if result is None:
                    break
                pickle.dump(result, f)  # 逐条写入 pkl 文件
        logger.info(f"Results saved to {output_file}")

    @staticmethod
    def log_processed_file(pdb_file, log_file):
        with open(log_file, "a") as f:
            f.write(str(pdb_file) + "\n")

    @staticmethod
    def load_processed_files(log_file):
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                return set(Path(line.strip()) for line in f)
        return set()

    @classmethod
    def batch_process_pdb_directory(cls, pdb_directory, output_file, log_file):
        pdb_files = list(Path(pdb_directory).glob("*.pdb"))

        processed_files = cls.load_processed_files(log_file)

        total_tasks = len(pdb_files) - len(processed_files)

        if total_tasks == 0:
            logger.info("All files have been processed.")
            return
        else:
            logger.info(f"Total input: {len(pdb_files)}")
            logger.info(f"Processed tasks: {len(processed_files)}")
            logger.info(f"TODO tasks: {total_tasks}")

        # share queue
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        # start consumer thread
        writer_process = multiprocessing.Process(target=cls.file_writer, args=(queue, output_file))
        writer_process.start()

        # get available cpu cores
        num_workers = multiprocessing.cpu_count()
        logger.info(f"Starting {num_workers} producer threads and 1 consumer thread.")

        # producer threads pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=total_tasks) as pbar:
                futures = {
                    executor.submit(cls.process_pdb_file, pdb_file, queue, processed_files, log_file): pdb_file
                    for pdb_file in pdb_files if pdb_file not in processed_files
                }
                for future in concurrent.futures.as_completed(futures):
                    pdb_file = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"{pdb_file} generated an exception: {e}")
                    pbar.update(1)  # update tqdm bar
        logger.info("All producer threads have been closed safely.")

        # send end signal to consumer
        queue.put(None)
        writer_process.join()  # wait consumer thread to finish
        logger.info("Consumer thread closed safely.")



if __name__ == '__main__':
    pdb_dir = r"/Users/douzhixin/Developer/qPacking/code/data/raw"
    output_pkl_file = r"/Users/douzhixin/Developer/qPacking/code/data/results.pkl"
    log_file = r"/Users/douzhixin/Developer/qPacking/code/data/processed.log"
    Analyzer.batch_process_pdb_directory(pdb_dir, output_pkl_file, log_file)
