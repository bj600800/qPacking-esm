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
import subprocess
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
import matplotlib.colors as mcolors

from qpacking.hydrocluster import cluster_extractor
from qpacking.utils import logger
logger = logger.setup_log(name=__name__)



class Analyzer:
    def __init__(self, cluster_graphs, structure, pdb_file):
        self.pdb_file = pdb_file
        self.cluster_graphs = cluster_graphs
        self.structure = structure

    def draw_graph(self, G):
        pos = nx.spring_layout(G)  # 生成布局，使节点分布合理

        # 获取边
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # None 用于断开线段
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color="gray"),
            hoverinfo="none",
            mode="lines"
        )

        # 获取节点
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"residue {node} (度: {G.degree[node]})")  # 鼠标悬停显示度数

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            textposition="top center",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                size=15,
                color="lightblue",
                line=dict(width=2, color="black")
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        fig.show()


    def draw_graph_interactive(self, G, output_file="graph.html"):
        net = Network(notebook=True, directed=False, cdn_resources='in_line')
        G = nx.relabel_nodes(G, lambda x: str(x))
        net.from_nx(G)  # 载入 networkx 图

        # 获取最大度数，用于归一化
        max_degree = max(dict(G.degree).values()) if G.number_of_nodes() > 0 else 1

        # 定义颜色映射函数
        def get_color(degree):
            base_color = mcolors.hex2color("#0d47a1")
            min_color = [c + (1 - c) for c in base_color]
            factor = degree / max_degree  # 归一化
            color = [(1 - factor) * min_c + factor * base_c for min_c, base_c in zip(min_color, base_color)]
            return mcolors.to_hex(color)

        # 设置节点样式
        for node in G.nodes():
            degree = G.degree[node]
            net.get_node(node)["color"] = get_color(degree)
            net.get_node(node)["size"] = 15
            net.get_node(node)["label"] = f"residue {node} (degree: {degree})"

        # 生成 HTML 并打开
        net.show(output_file)
        print(f"交互式图已生成：{output_file}")

    def show_hydrocluster_pymol(self):
        pymol_bin = 'pymol'
        cmd = []
        for i, cluster in enumerate(self.cluster_graphs):
            command = f"sele Cluster {i}, res {'+'.join([str(res_id) for res_id in cluster.nodes()])}"
            cmd.append(command)
        pymol_cmd = '; '.join(cmd)

        # get Anaconda base environment path
        conda_base = r"/opt/anaconda3"
        pymol_path = os.path.join(conda_base, "bin")

        # add base pymol dir to the current environmental variable path
        os.environ["PATH"] = pymol_path + os.pathsep + os.environ["PATH"]

        command_args = [pymol_bin, self.pdb_file, '-d', pymol_cmd]
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.communicate()


    def get_degree(self):

        for G in self.cluster_graphs:
            degree_dict = dict(G.degree())
            # 打印每个节点的度
            for node, degree in degree_dict.items():
                print(f"residue {node} degree：{degree}")

            self.draw_graph_interactive(G)
            input()
        self.show_hydrocluster_pymol()

    def calc_contact_count(self):
        pass

    def run(self):
        self.get_degree()
        self.calc_contact_count()


        # clean cache
        self.cluster_graphs = None
        self.structure = None

        return None

    @staticmethod
    def file_writer(queue, output_file):
        """consumer: write pkl file with FIFO"""
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

    @staticmethod
    def process_pdb_file(pdb_file, queue, processed_log, log_file):
        """producer for one file"""
        if pdb_file in processed_log:
            return

        cluster_graphs, structure = cluster_extractor.run(pdb_file)
        analyzer = Analyzer(cluster_graphs, structure, pdb_file)
        result = analyzer.run()

        # FIFO parallel processing: put result into queue
        queue.put((pdb_file, result))  # 结果入队列

        # Task logging
        Analyzer.log_processed_file(pdb_file, log_file)  # 记录到日志文件



    @classmethod
    def batch_process_pdb_directory(cls, pdb_directory, output_file, log_file):
        # get available cpu cores
        num_workers = multiprocessing.cpu_count()

        #TODO: Remove this debug block before production
        if os.path.exists(log_file):
            os.remove(log_file)
        num_workers = 1
        #TODO: Remove this debug block before production

        logger.info(f"Starting {num_workers} producer threads and 1 consumer thread.")

        pdb_files = list(Path(pdb_directory).glob("*.pdb"))[:1]
        processed_files = cls.load_processed_files(log_file)
        total_tasks = len(pdb_files) - len(processed_files)

        if total_tasks == 0:
            logger.info("All files have been processed.")
            # return
        else:
            logger.info(f"Total input: {len(pdb_files)}")
            logger.info(f"Processed tasks: {len(processed_files)}")
            logger.info(f"TODO tasks: {total_tasks}")

        # share queue
        queue = multiprocessing.Queue()

        # start consumer thread
        writer_process = multiprocessing.Process(target=cls.file_writer, args=(queue, output_file))
        writer_process.start()


        # producer threads pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=total_tasks, desc="Batch processing") as pbar:
                futures = {
                    executor.submit(cls.process_pdb_file, pdb_file, queue, processed_files, log_file): pdb_file
                    for pdb_file in pdb_files if pdb_file not in processed_files
                }
                for future in concurrent.futures.as_completed(futures):
                    pdb_file = futures[future]

                    try:
                        future.result()
                    except Exception as e:
                        raise
                        logger.error(e)
                    pbar.update(1)  # update tqdm bar
        executor.shutdown(wait=True)
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

