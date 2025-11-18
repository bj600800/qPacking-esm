"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/15

# Description: Test fine-tuned ESM-2 model for predicting hydrophobic features
# ------------------------------------------------------------------------------
"""
import torch
from transformers import EsmTokenizer
from safetensors.torch import load_file

import torch.nn.functional as F
# === 参数 ===
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import PeftModel, PeftConfig
from qpacking_esm.model.heads import ClassificationHead
# ==== 1. 配置路径 ====
best_model_path = "/Users/douzhixin/Developer/qPacking-esm/data/test/checkpoints/hydrophobic_binary/best"  # 修改为你保存模型的目录

# ==== 2. 加载 tokenizer ====
tokenizer = EsmTokenizer.from_pretrained(best_model_path)

# 1️⃣ 加载 backbone + LoRA
peft_config = PeftConfig.from_pretrained(best_model_path)
base_model = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
backbone = PeftModel.from_pretrained(base_model, best_model_path)

# 2️⃣ 复用训练时的分类头
class TokenClassificationModel(nn.Module):
    def __init__(self, backbone, classifier_head_path):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(backbone.config.hidden_size, 2)
        self.head.classifier.load_state_dict(torch.load(classifier_head_path, weights_only=True))

    def forward(self, input_ids, attention_mask=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(hidden)
        return logits


model = TokenClassificationModel(backbone, f"{best_model_path}/classifier_head.pt")
model.eval()

# ==== 6. 输入示例序列 ====
sequences = ["ERTFIAIKPDGVQRGLVGEIIKRFEQKGFRLVAMKFLASEEHLKQHYIDLKDRPFFPGLVKYMNSGPVVAMVWEGLVVKTGRVMLGETNPADSKPGTIRGDFCIQVGRNIIHGSDSVKSAEKEISLWFKPEELVDYKCAHDWVYE"]
encoded = tokenizer(sequences, padding=True, return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# ==== 7. 模型推理 ====
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(logits.logits, dim=-1)

# ==== 8. 输出结果 ====
print("Predicted labels:", predictions.tolist()[0][1:-1])

res_type = []
res_ids = []
hydrophobic_aa = ['V', 'I', 'L', 'A', 'M']
for i, label in enumerate(predictions.tolist()[0][1:-1]):
    if label == 1:
        res_ids.append(i + 1)
        res_type.append(sequences[0][i])
print('sele res '+'+'.join([str(i) for i in res_ids]))
print(res_type)
seq_len = len(sequences[0])
print(seq_len)

hydrophobic_aa_all_count = 0
for i in sequences[0]:
    if i in hydrophobic_aa:
        hydrophobic_aa_all_count += 1

print(hydrophobic_aa_all_count)
print(len(res_type))