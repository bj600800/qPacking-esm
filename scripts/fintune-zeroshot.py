"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/15

# Description: 
# ------------------------------------------------------------------------------
"""
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import PeftModel

from sklearn.metrics import classification_report
from qpacking.models import dataset
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)

class TokenClassificationModel(nn.Module):
    def __init__(self, model_with_lora, num_clusters):
        super().__init__()
        self.model = model_with_lora
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_clusters)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(hidden_states)
        return logits


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [B, L, C]
            predictions = torch.argmax(logits, dim=-1)  # [B, L]

            # Flatten mask
            mask = labels != -100  # ignore index
            preds_flat = predictions[mask]
            labels_flat = labels[mask]

            all_preds.extend(preds_flat.cpu().tolist())
            all_labels.extend(labels_flat.cpu().tolist())
    logger.info("标签类别分布:", Counter(all_labels))
    logger.info("预测类别分布:", Counter(all_preds))
    logger.info("\n Classification Report:")
    logger.info(classification_report(all_labels, all_preds, digits=4))

model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t30_150M_UR50D"
peft_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/qpacking/best"

# 1. 先加载基础模型
base_model = EsmModel.from_pretrained(model_dir, add_pooling_layer=False)
base_model.enable_input_require_grads()
# 其他配置
seed = 3407
test_ratio = 0.1
batch_size = 8
# 2. 直接用 PeftModel.from_pretrained 加载带 LoRA 权重的模型
# model_with_lora = PeftModel.from_pretrained(base_model, peft_dir, is_trainable=True)

# 3. 包裹分类头
# model = TokenClassificationModel(model_with_lora, num_clusters=5)
model = TokenClassificationModel(base_model, num_clusters=5)
model.eval()

# 加载 tokenizer（对应 ESM 模型）
tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)

fasta_file = r"/Users/douzhixin/Developer/qPacking/data/test/sequence.fasta"
pkl_file = r"/Users/douzhixin/Developer/qPacking/data/test/class_results.pkl"
tokenized_cache_path = r"/Users/douzhixin/Developer/qPacking/data/test/tokenized_cache"
_, val_loader, num_clusters = dataset.run(
        fasta_file,
        pkl_file,
        model_dir,
        tokenized_cache_path,
        test_ratio,
        batch_size,
        seed,
    )

device = 'cpu'
evaluate(model, val_loader, device)

