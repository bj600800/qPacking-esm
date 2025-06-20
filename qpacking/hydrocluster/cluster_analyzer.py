"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/3/12

# Description: This script calculates various Hydrophobic Packing Descriptors (HPDs) from PDB files in batch mode.
# 1. Extract hydrophobic clusters from a PDB file.
# 2. Calculate different metrics for each cluster at the residue level.
# 3. Run batch calculations for all PDB files in a directory.
# 4. save raw data to a pickle file -> results.pkl
# ------------------------------------------------------------------------------
"""
import os
import pickle
import concurrent.futures
import multiprocessing
import time
import humanize

from tqdm import tqdm
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

from biotite.structure import sasa, centroid
from qpacking.hydrocluster import cluster_identifier
from qpacking.utils import visualization
from qpacking.utils.logger import setup_log
logger = setup_log(name=__name__, enable_file_log=False)


class Analyzer:
    def __init__(self, cluster_graphs, structure, pdb_file, dssp):
        self.pdb_file = pdb_file
        self.cluster_graphs = cluster_graphs
        self.structure = structure
        self.first_res_id = structure.res_id[0]
        self.dssp = dssp

    def get_class(self, G, i):
        node_labels = nx.get_node_attributes(G, 'res_name')
        class_feature = {}
        for res_id, _ in node_labels.items():
            res_id = int(res_id) - int(self.first_res_id)
            class_feature[res_id] = i  # cluster index starts from 0, -100 stands for mask of non-cluster residues

        return class_feature

    def get_area(self, G):
        node_labels = nx.get_node_attributes(G, 'res_name')
        area_feature = {}
        for res_id, res_name in node_labels.items():
            residue_mask = np.isin(self.structure.res_id, res_id)
            res_array = self.structure[residue_mask]
            res_id = int(res_id) - int(self.first_res_id)
            area_feature[res_id] = sum(sasa(res_array))

        return area_feature

    def get_degree(self, G):
        degree_feature = {}
        degree_dict = dict(G.degree())
        for node, degree in degree_dict.items():
            node = int(node) - int(self.first_res_id)
            degree_feature[node] = degree

        # visualization.draw_graph_interactive(G, "degree_graph.html")
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
        dssp = DSSP(model, self.pdb_file, dssp=self.dssp)

        # create dssp rasa dict
        dssp_rasa_dict = {
            key[1][1]: dssp[key][3]
            for key in dssp.keys()
        }

        # get rasa
        rasa_dict = {
            int(res) - int(self.first_res_id): dssp_rasa_dict.get(res, None)
            for res in packing_res
        }

        return rasa_dict

    def get_packing_order(self, G):
        """
        packing order (PO) defines the difficulties of each hydrophobic residue in the cluster.
        The more difficult to pack, the higher the order.
        cite: https://doi.org/10.1006/jmbi.1998.1645
        :return: {res_id: PO} dictionary
        """
        # visualization.show_hydrocluster_pymol(self.pdb_file, self.cluster_graphs)
        # input()
        hydro_res = G.nodes()
        contact = G.edges()
        N_contact = len(list(contact))
        l_cluster = len(list(hydro_res))
        Nl_multiply = N_contact * l_cluster

        order_dict = {}
        for res in hydro_res:
            s_pair = 0
            for pair in contact:
                if res in pair:
                    s_pair += abs(int(pair[0]) - int(pair[1]))
            res = int(res) - int(self.first_res_id)
            order_dict[res] = s_pair / Nl_multiply
        return order_dict


    def get_centrality(self, G):
        centrality_dict = {}
        res_list = list(G.nodes())
        coords = []
        for res_id in res_list:
            residue_mask = np.isin(self.structure.res_id, res_id) & np.isin(self.structure.atom_name, 'CA')
            coord = self.structure[residue_mask].coord
            coords.append(coord[0])
        D = cdist(np.array(coords), np.array(coords))  # Matrix NxN
        distance_sums = np.sum(D, axis=1)
        N = len(res_list) - 1

        for i in range(len(res_list)):
            res_list[i] = int(res_list[i]) - int(self.first_res_id)
            centrality_dict[res_list[i]] = N / distance_sums[i]
        return centrality_dict


    def run(self):
        packing_res = []
        struct_features = {'class':{},
                           'area': {},
                           'degree': {},
                           'rsa': {},
                           'order': {},
                           'centrality': {}}
        try:
            for i, G in enumerate(self.cluster_graphs):
                class_feature = self.get_class(G, i)
                area_feature = self.get_area(G)
                degree_feature = self.get_degree(G)
                order_feature = self.get_packing_order(G)
                centrality_feature = self.get_centrality(G)

                packing_res.extend(list(G.nodes()))
                struct_features['class'].update(class_feature)
                struct_features['area'].update(area_feature)
                struct_features['degree'].update(degree_feature)
                struct_features['order'].update(order_feature)
                struct_features['centrality'].update(centrality_feature)
            rsa_dict = self.get_rasa(packing_res)
            struct_features['rsa'] = rsa_dict

        except Exception as e:
            logger.warning(f"Error in calculating {self.pdb_file}: {e}")
            return False

        # clean cache
        self.cluster_graphs = None
        self.structure = None
        if all(v for v in struct_features.values()):
            return struct_features
        else:
            return False


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
    def file_writer(queue, output_file, buffer_size=10000):  # buffer size too small will not be enough for the last time writing.
        """
        Consumer thread to write all results (new and old) to a file.
        Each buffer was written once for each result.
        :param queue:
        :param output_file: pkl
        :param buffer_size:
        :return:
        """
        # Load existing results before updating
        results = Analyzer.load_existing_results(output_file)
        if results is None:
            results = {}

        buffer = []
        while True:
            result = queue.get()
            if result is None:
                break
            try:
                if isinstance(result, dict):
                    buffer.append(result)
                    if len(buffer) >= buffer_size:
                        results.update({k: v for d in buffer for k, v in d.items()})
                        logger.info("Writing buffer to file...")
                        with open(output_file, "wb") as f:
                            pickle.dump(results, f)
                        buffer.clear()
                        logger.info("Writing complete.")
                else:
                    logger.error(f"Received object is not a dictionary: {result}")
            except Exception as e:
                logger.error(f"Error processing {result}: {e}")
                continue
        try:
            if buffer:
                results.update({k: v for d in buffer for k, v in d.items()})

                logger.info("Writing the remaining buffer to file...")
                with open(output_file, "wb") as f:
                    pickle.dump(results, f)
                logger.info("Writing complete.")

        except Exception as e:
            logger.error(f"Error during final buffer write: {e}")

        # Final report
        if results:
            try:
                feature_types = list(list(results.values())[0].keys())
                feature_num = len(feature_types)
                logger.info(
                    f"A total of {feature_num} types of structural features were collected: {', '.join(feature_types)}")
            except Exception as e:
                logger.warning(f"Final summary error: {e}")
        else:
            logger.warning("No valid structural features were collected.")

    @classmethod
    def process_pdb_file(cls, pdb_file, queue, dssp):
        cluster_graphs, structure = cluster_identifier.run(pdb_file)
        analyzer = cls(cluster_graphs, structure, pdb_file, dssp)
        result = analyzer.run()
        # FIFO
        pdb_name = pdb_file.stem

        if result:
            # Only put valid data in there
            queue.put({pdb_name: result})

    @classmethod
    def batch_process_pdb_files(cls, pdb_directory, output_pkl_file, dssp):
        time1 = time.time()
        # num_workers = multiprocessing.cpu_count()
        num_workers = 2  # It is the best performance
        logger.info(f"Starting {num_workers} producer threads and 1 consumer thread.")

        #TODO: Remove this debug block before production
        # if os.path.exists(output_pkl_file):
        #     os.remove(output_pkl_file)

        #TODO: Remove this debug block before production

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

        # send The End signal to consumer
        queue.put(None)
        writer_process.join()  # wait consumer thread to finish

        time2 = time.time()
        time_cost = time2 - time1
        time_str = humanize.naturaldelta(time_cost)

        logger.info("Consumer thread closed safely.")
        logger.info(f"All threads closed and {len(pdb_files)} structure features saved to {output_pkl_file}")
        logger.info(f"Time cost: {time_str}")

        ## TEST: print the saved results in pkl file.
        # load_existing_results = Analyzer.load_existing_results(output_pkl_file)
        # print(load_existing_results)


if __name__ == '__main__':
    pdb_dir = r"/Users/douzhixin/Developer/qPacking/data/test/structure"
    output_pkl_file = r"/Users/douzhixin/Developer/qPacking/data/test/results.pkl"
    dssp = "mkdssp"
    Analyzer.batch_process_pdb_files(pdb_dir, output_pkl_file, dssp)
