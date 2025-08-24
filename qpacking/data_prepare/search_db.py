# ------------------------------------------------------------------------------
# Author:    Dou zhixin
# Email:     bj600800@gmail.com
# DATE:      2024/11/18
#
# Description: Searching ted sql dataset based on the required conditions.
# ------------------------------------------------------------------------------
import sqlite3
import argparse

from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

#### ARGUMENTS PARSER ####
parser = argparse.ArgumentParser(description='get pdbs')
parser.add_argument('--db', required=True, help='sqldb file')
parser.add_argument('--id', required=True, help='output id file')
parser.add_argument('--helix', required=False, help='helix num')
parser.add_argument('--strand', required=False, help='strand num')
parser.add_argument('--turn', required=False, help='turn num')
parser.add_argument('--nres', required=False, help='residue num')
args = parser.parse_args()
#### END OF ARGUMENTS PARSER ####


def get_ted_ids(sql_db, output_id_file, helix, strand, turn, nres):
    logger.info(f"Start database searching...")
    conn = sqlite3.connect(sql_db)
    cursor = conn.cursor()

    # define conditions
    plddt_threshold = 90
    consensus_level = 'high'
    pack_density_threshold = 10.333
    norm_rg_threshold = 0.356
    num_helix = helix if helix is not None else None
    num_strand = strand if strand is not None else None
    num_turn = turn if turn is not None else None
    num_res = nres if nres is not None else None

    # conditional sqlite query
    query = """
            SELECT ted_id, chopping
            FROM summary
            WHERE INSTR(chopping, '_') = 0
                AND plddt >= ?
                AND consensus_level = ?
                AND packing_density >= ?
                AND norm_rg < ?
        """

    params = [plddt_threshold, consensus_level, pack_density_threshold, norm_rg_threshold]

    if num_helix is not None:
        query += " AND num_helix >= ?"
        params.append(num_helix)

    if num_strand is not None:
        query += " AND num_strand >= ?"
        params.append(num_strand)

    if num_turn is not None:
        query += " AND num_turn >= ?"
        params.append(num_turn)

    if num_res is not None:
        num_res = num_res.split(',')
        min_res, max_res = map(int, num_res)
        query += " AND nres_domain >= ? AND nres_domain <= ?"
        params.append(min_res)
        params.append(max_res)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    with open(output_id_file, 'w', ) as f:
        for i in rows:
            f.write('\t'.join(i) + '\n')
    logger.info(f"Saved total {len(rows)} items to {output_id_file}")


if __name__ == '__main__':
    sql_db = args.db
    output_id_file = args.id
    helix = args.helix
    strand = args.strand
    turn = args.turn
    nres = args.nres
    get_ted_ids(sql_db, output_id_file, helix, strand, turn, nres)
