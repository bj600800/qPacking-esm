# ------------------------------------------------------------------------------
# Author:    Dou zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/4
#
# Description: 
# ------------------------------------------------------------------------------
import os
import requests
import argparse
from requests.adapters import Retry
from tqdm import tqdm
import subprocess

from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

#### ARGUMENTS PARSER ####
parser = argparse.ArgumentParser(description='get pdbs')
parser.add_argument('--id', required=True, help='output id file')
parser.add_argument('--pdb', required=True, help='output pdb dir')

args = parser.parse_args()

#### END OF ARGUMENTS PARSER ####


def get_ted_ids(output_id_file):
    if os.path.exists(output_id_file):
        with open(output_id_file, "r") as f:
            id_dict = {line.split('\t')[0].strip(): line.split('\t')[1].strip() for line in f}

        logger.info(f"Found {len(id_dict)} existing items in {output_id_file}")

    else:
        logger.error(f"No id files")
        exit(1)
    return id_dict


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

    uniprot_id = ted_id.split('-')[1]
    api_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    session = _start_request_session()
    response = session.get(api_url, headers=headers)
    if response.status_code == 200:
        pdb_string = response.text
        return pdb_string

    else:
        logger.warning(f"Seq2Struct uniprot ID failed: {ted_id}")


def crawl_struct(id_dict, structure_folder):
    if not os.path.exists(structure_folder):
        os.mkdir(structure_folder)
        
    def search_exist_struct(structure_folder):
        exist_structure = [file for file in os.listdir(structure_folder)
                           if os.path.isfile(os.path.join(structure_folder, file))
                           and os.path.getsize(os.path.join(structure_folder, file)) > 0]
        return [os.path.splitext(item)[0] for item in exist_structure]

    list_exist_item = set(search_exist_struct(structure_folder))
    ids = set(id_dict.keys())

    continue_id = ids-list_exist_item
    
    logger.info(f"Existance: {len(list_exist_item)}, All: {len(ids)}, Continue: {len(continue_id)}")

    if continue_id:
        for i in tqdm(continue_id, total=len(continue_id)):
            res_pos = id_dict[i].replace('-', ':')
            pdb_string = get_struct(i)
            if pdb_string:
                save_path = os.path.join(structure_folder, i + ".pdb")
                with open(save_path, "w") as f:
                    f.write(pdb_string)
                try:
                    command = ['pdb_selres', f'-{res_pos}', save_path]
                    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with open(save_path, "w") as f:
                        f.write(result.stdout.decode())
                except:
                    pass

    logger.info(f"Crawled structures: {len(os.listdir(structure_folder))}")


structure_folder = args.pdb
output_id_file = args.id
id_dict = get_ted_ids(output_id_file)
crawl_struct(id_dict, structure_folder)
