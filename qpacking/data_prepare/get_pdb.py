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
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[408, 429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
    return session


def get_struct(ted_id):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "From": "bj600800@gmail.com"  # ALLWAYS TELLs WHO YOU ARE
    }

    uniprot_id = ted_id.split('-')[1]
    api_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    session = _start_request_session()
    response = session.get(api_url, headers=headers, timeout=10)
    if response.status_code == 200:
        pdb_string = response.text
        return pdb_string

    else:
        logger.warning(f"Seq2Struct uniprot ID failed: {ted_id}")


def crawl_struct(id_dict, structure_folder):
    full_length_dir = os.path.join(structure_folder, 'raw')
    os.makedirs(full_length_dir, exist_ok=True)

    domain_dir = os.path.join(structure_folder, 'domain')
    os.makedirs(domain_dir, exist_ok=True)

    def search_exist_struct(full_length_dir):
        exist_structure = [file for file in os.listdir(full_length_dir)
                           if os.path.isfile(os.path.join(full_length_dir, file))
                           and os.path.getsize(os.path.join(full_length_dir, file)) > 0]
        return [os.path.splitext(item)[0] for item in exist_structure]

    list_exist_item = set(search_exist_struct(full_length_dir))
    ids = set(id_dict.keys())

    continue_id = ids-list_exist_item
    
    logger.info(f"Existance: {len(list_exist_item)}, All: {len(ids)}, Continue: {len(continue_id)}")

    if continue_id:
        for i in tqdm(continue_id, total=len(continue_id)):
            res_pos = id_dict[i].replace('-', ':')
            pdb_string = get_struct(i)
            if pdb_string:
                full_length_path = os.path.join(full_length_dir, i + ".pdb")
                with open(full_length_path, "w") as f:
                    f.write(pdb_string)
                try:
                    command = ['pdb_selres', f'-{res_pos}', full_length_path]
                    domain_path = os.path.join(domain_dir, i + ".pdb")
                    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with open(domain_path, "w") as f:
                        f.write(result.stdout.decode())
                except:
                    with open(os.path.join(structure_folder, 'domain_error.log'), "w+") as f:
                        f.write(i+'\n')

    logger.info(f"Crawled structures: {len(os.listdir(structure_folder))}")


if __name__ == '__main__':
    structure_folder = args.pdb
    output_id_file = args.id
    id_dict = get_ted_ids(output_id_file)
    crawl_struct(id_dict, structure_folder)

