"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/3/12

# Description: This script calculates various metrics for hydrophobic clusters from PDB files in batch mode.
# 1. Extract hydrophobic clusters from a PDB file.
# 2. Calculate different metrics for each cluster at the residue level.
# 3. Run batch calculations for all PDB files in a directory.
# 4. save raw data to a pickle file -> results.pkl
# ------------------------------------------------------------------------------
"""
import os
import pickle
import json
import concurrent.futures
import multiprocessing

from tqdm import tqdm
from pathlib import Path
import networkx as nx
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

from biotite.structure.info import vdw_radius_single
from qpacking.hydrocluster import cluster_identifier
from qpacking.utils import visualization
from qpacking.utils.logger import setup_log
logger = setup_log(name=__name__, enable_file_log=False)


class Analyzer:
    def __init__(self, cluster_graphs, structure, pdb_file, dssp):
        self.pdb_file = pdb_file
        self.cluster_graphs = cluster_graphs
        self.structure = structure
        self.dssp = dssp
    @staticmethod
    def resi_area():
        r_vdw = vdw_radius_single("C")
        pi = np.pi
        ile_area = 4 * 4 * pi * r_vdw ** 2
        leu_area = 4 * 4 * pi * r_vdw ** 2
        val_area = 3 * 4 * pi * r_vdw ** 2
        area = {'ILE': round(ile_area,2), 'LEU': round(leu_area,2), 'VAL': round(val_area,2)}
        return area

    @staticmethod
    def get_area(G, res_area_dict):
        node_labels = nx.get_node_attributes(G, 'res_name')
        area_feature = {str(res_id): res_area_dict[res_name] for res_id, res_name in node_labels.items()}
        return area_feature

    @staticmethod
    def get_degree(G):
        degree_feature = {}
        degree_dict = dict(G.degree())
        for node, degree in degree_dict.items():
            degree_feature[str(node)] = degree
        return degree_feature

    def get_rasa(self, packing_res):
        """
        Calculate the relative solvent accessibility (RSA) for a list of residues using biopython-dssp-rasa.
        :param resi_list: List of residue indices
        :return: {res_i: rsa} dictionary
        """
        p = PDBParser()
        structure = p.get_structure("protein", self.pdb_file)
        model = structure[0]
        dssp = DSSP(model, self.pdb_file, dssp='mkdssp')

        # create dssp rasa dict
        dssp_rasa_dict = {
            key[1][1]: dssp[key][3]
            for key in dssp.keys()
        }

        # get rasa
        rasa_dict = {
            res: dssp_rasa_dict.get(res, None)
            for res in packing_res
        }

        return rasa_dict

    def run(self):
        packing_res = []
        struct_features = {'area': {}, 'degree': {}, 'rsa': {}}
        res_area_dict = self.resi_area()
        # visualization.show_hydrocluster_pymol(self.pdb_file, self.cluster_graphs)

        for G in self.cluster_graphs:
            area_feature = self.get_area(G, res_area_dict)
            degree_feature = self.get_degree(G)
            packing_res.extend(list(G.nodes()))

            struct_features['area'].update(area_feature)
            struct_features['degree'].update(degree_feature)

        rsa_dict = self.get_rasa(packing_res)
        struct_features['rsa'] = rsa_dict



        # clean cache
        self.cluster_graphs = None
        self.structure = None
        return struct_features

    @staticmethod
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
                    logger.error(f"Loaded object is not a dictionary: {type(results_dict)}")
                    return {}
                return results_dict
        except (FileNotFoundError, EOFError):
            return {}
        except Exception as e:
            logger.error(f"An error occurred while loading pickle file: {e}")
            return {}

    @staticmethod
    def file_writer(queue, output_file, buffer_size=10000):
        """
        Consumer thread to write all results (new and old) to a file.
        Each buffer write once for each result.
        :param queue:
        :param output_file: pkl
        :param buffer_size:
        :return:
        """
        # Load existing results before updating
        results = Analyzer.load_existing_results(output_file)  # persistence variable


        buffer = []
        while True:
            result = queue.get()
            if result is None:
                break
            elif isinstance(result, dict):
                buffer.append(result)
                if len(buffer) >= buffer_size:
                    results.update({k: v for d in buffer for k, v in d.items()})
                    with open(output_file, "wb") as f:
                        pickle.dump(results, f)
                    buffer.clear()
            else:
                logger.error(f"Received object is not a dictionary: {result}")

        if buffer:
            results.update({k: v for d in buffer for k, v in d.items()})
            with open(output_file, "wb") as f:
                pickle.dump(results, f)
        feat_num = len(list(results.values())[0].keys())

        logger.info(f"Total types of structure feature collected: {feat_num}")

    @classmethod
    def process_pdb_file(cls, pdb_file, queue, dssp):
        cluster_graphs, structure = cluster_identifier.run(pdb_file)
        analyzer = cls(cluster_graphs, structure, pdb_file, dssp)
        result = analyzer.run()
        # FIFO parallel processing: put result into queue
        pdb_name = pdb_file.stem

        queue.put({pdb_name: result})

    @classmethod
    def batch_process_pdb_files(cls, pdb_directory, output_pkl_file, dssp):
        num_workers = multiprocessing.cpu_count()

        #TODO: Remove this debug block before production
        if os.path.exists(output_pkl_file):
            os.remove(output_pkl_file)
        num_workers = 1
        logger.info("workers 1 for debug.")
        #TODO: Remove this debug block before production

        logger.info(f"Starting {num_workers} producer threads and 1 consumer thread.")

        pdb_files = list(Path(pdb_directory).glob("*.pdb"))
        logger.info(f"Found {len(pdb_files)} PDB files in {Path(pdb_directory)}")
        if os.path.exists(output_pkl_file):
            existing_results = list(cls.load_existing_results(output_pkl_file).keys())
            logger.info(f"Existing results loaded from {output_pkl_file}: {len(existing_results)}")
        else:
            existing_results = {}

        tasks = [i for i in pdb_files if i.stem not in existing_results]
        task_count = len(tasks)

        if task_count == 0:
            logger.info("All files have been processed.")
            # load_existing_results = Analyzer.load_existing_results(output_pkl_file)
            # json_output = json.dumps(load_existing_results, indent=1)
            # print(json_output)
            return
        else:
            logger.info(f"Task number: {task_count}")

        queue = multiprocessing.Queue()

        # start consumer thread
        writer_process = multiprocessing.Process(target=cls.file_writer, args=(queue, output_pkl_file))
        writer_process.start()

        # start producer threads pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=task_count, desc="Batch processing") as pbar:
                futures = {
                    executor.submit(cls.process_pdb_file, pdb_file, queue, dssp): pdb_file
                    for pdb_file in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    pdb_file = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(str(pdb_file)+': '+str(e))
                    pbar.update(1)  # update tqdm bar

        executor.shutdown(wait=True)
        logger.info("Producer threads closed safely.")

        # send end signal to consumer
        queue.put(None)
        writer_process.join()  # wait consumer thread to finish
        logger.info("Consumer thread closed safely.")

        logger.info(f"All threads closed and {len(pdb_files)} structure features saved to {output_pkl_file}")

        # print the saved results in pkl file.
        # load_existing_results = Analyzer.load_existing_results(output_pkl_file)
        # json_output = json.dumps(load_existing_results, indent=1)
        # print(json_output)


if __name__ == '__main__':
    pdb_dir = r"/Users/douzhixin/Developer/qPacking/code/data/raw"
    output_pkl_file = r"/Users/douzhixin/Developer/qPacking/code/data/results.pkl"
    dssp = "mkdssp"
    Analyzer.batch_process_pdb_files(pdb_dir, output_pkl_file, dssp)
