# ------------------------------------------------------------------------------
# Author:    Dou zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/4
#
# Description: 
# ------------------------------------------------------------------------------
import sqlite3
import os
import csv
import requests
from requests.adapters import Retry
from tqdm import tqdm

from qpacking.utilities import logger

logger = logger.setup_log(name=__name__)


def _start_request_session():
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[408, 429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
    return session


def get_struct(ted_id):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400",
        "From": "bj600800@gmail.com"  # ALLWAYS TELLs WHO YOU ARE
    }

    api_url = f"https://ted.cathdb.info/api/v1/files/{ted_id}.pdb"
    session = _start_request_session()
    response = session.get(api_url, headers=headers)
    if response.status_code == 200:
        pdb_string = response.text
        return pdb_string

    else:
        logger.warning(f"Seq2Struct uniprot ID failed: {ted_id}")


def get_ted_ids(sql_db, output_id_file):
    conn = sqlite3.connect(sql_db)
    cursor = conn.cursor()

    # define conditions
    cath_label = '3.20.20.70'
    plddt_threshold = 90
    consensus_level = 3
    pack_density_threshold = 0.333
    norm_rg_threshold = 0.356

    # sqlite query
    query = """
    SELECT ted_id
    FROM your_table
    WHERE cath_label = ? AND plddt >= ? AND consensus_level = ? AND pack_density >= ? AND norm_rg < ?
    """
    cursor.execute(query, (cath_label, plddt_threshold, consensus_level, pack_density_threshold, norm_rg_threshold))
    rows = [i[0] for i in cursor.fetchall()]
    cursor.close()
    conn.close()

    with open(output_id_file, 'w', ) as f:
        for i in rows:
            f.write(i+'\n')

    print(f"数据已成功保存到 {output_id_file}")
    print(len(rows))
    input()
    return rows


def crawl_struct(id_list, structure_folder):


    def search_exist_struct(structure_folder):
        exist_structure = [file for file in os.listdir(structure_folder)
                           if os.path.isfile(os.path.join(structure_folder, file))
                           and os.path.getsize(os.path.join(structure_folder, file)) > 0]
        return [os.path.splitext(item)[0] for item in exist_structure]

    id_list = list(set(id_list) - set(search_exist_struct(structure_folder)))

    if id_list:
        for ted_id in tqdm(id_list):
            pdb_string = get_struct(ted_id)
            if pdb_string:
                save_path = os.path.join(structure_folder, ted_id + ".pdb")
                with open(save_path, "w") as f:
                    f.write(pdb_string)

    logger.info(f"Crawled structures: {len(os.listdir(structure_folder))}")

sql_db = r"/home/u2600215/qpacking/data/ted/ted.db"
structure_folder = r"/home/u2600215/qpacking/data/ted/pdb"
output_id_file = r"/home/u2600215/qpacking/data/ted/id.txt"
id_list = get_ted_ids(sql_db, output_id_file)
crawl_struct(id_list, structure_folder)
