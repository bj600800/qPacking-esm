"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/2/14

# Description: train qPacking
# ------------------------------------------------------------------------------
"""
from qpacking.models import dataset
from qpacking.models.setup_train import train_cluster_classification

model_dir = "/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t30_150M_UR50D"
checkpoints_dir = "/Users/douzhixin/Developer/qPacking/code/checkpoints/qpacking"
logging_dir = "/Users/douzhixin/Developer/qPacking/code/logs"
TOKENIZED_CACHE_PATH = r"/Users/douzhixin/Developer/qPacking/data/test/tokenized_cache"
fasta_file = "/Users/douzhixin/Developer/qPacking/data/test/sequence.fasta"
pkl_file = "/Users/douzhixin/Developer/qPacking/data/test/class_results.pkl"

batch_size = 16
num_epochs = 200
seed = 3407
lr = 2e-5
lora_rank = 8
lora_alpha = 8
lora_dropout = 0.05
test_ratio = 0.1




train_dataloader, valid_dataloader, num_clusters = dataset.run(
    fasta_file, pkl_file, model_dir, TOKENIZED_CACHE_PATH, test_ratio, batch_size, seed
)
focal_gamma = 2.0
focal_alpha = None


train_cluster_classification(
    model_dir=model_dir,
    checkpoints_dir=checkpoints_dir,
    logging_dir=logging_dir,
    batch_size=batch_size,
    num_epochs=num_epochs,
    seed=seed,
    lr=lr,
    num_clusters=num_clusters,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    focal_gamma=focal_gamma,
    focal_alpha=focal_alpha
)
