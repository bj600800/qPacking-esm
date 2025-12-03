"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/7/10

# Description: 
# ------------------------------------------------------------------------------
"""
from transformers import EsmTokenizer, EsmModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from qpacking_esm.model.heads import ClassificationHead, RegressionHead
# ==== 1. 配置路径 ====
model_dir = "/Users/douzhixin/Developer/qPacking-esm/data/checkpoints/esm2_t30_150M_UR50D"  # 原始的 ESM2 模型名
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==== 2. 加载 tokenizer ====
tokenizer = EsmTokenizer.from_pretrained(model_dir)

# ==== 3. 加载原始 ESM2 模型（无微调） ====
base_model = EsmModel.from_pretrained(model_dir, add_pooling_layer=False)
base_model = base_model.to(device)

# ==== 4. 定义分类模型 ====
class TokenClassificationModel(nn.Module):
    def __init__(self, backbone, classifier_head_path):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(backbone.config.hidden_size, 2)

        # load state_dict
        state_dict = torch.load(classifier_head_path, map_location="cpu", weights_only=True)
        state_dict_linear = {k.replace("classifier.", ""): v for k, v in state_dict.items() if
                             k.startswith("classifier.")}
        self.head.classifier.load_state_dict(state_dict_linear)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(hidden)
        return logits


head_path = r"/Users/douzhixin/Developer/qPacking-esm/data/checkpoints/hpd/position/lora1/checkpoint-3500/task_head.pt"

# ==== 5. 初始化模型（不要加载微调头） ====
model = TokenClassificationModel(backbone=base_model, classifier_head_path=head_path).to(device)
model.eval()

# ==== 6. 输入序列 ====
sequences = ["VYGRNILNENGELVMLRGANRPGTEYCCVQYAKIFDGPHDQAQVDEMRKWKMNAVRVPLNEDCWLGVHAPETEYFGASYRKALEDYIKQYTDSNMAVIIDLHWASENGKLATQQIPMPNNGNSLLFWEDVARTFKDNSRVLFDLYNEPYPYGNSWANPDAWQCWKNGTDCGTLGYEASGMQQMIDAIRSTGSTNIILLSGIQYATSFTMFLDYQPTDPAKQMGVALHSYDFNYCRSRGCWDTYLKPVYSLFPMVATETGQKDCLHDFLSDFINYCDSNDIHYLAWSWLTGECGIPSLIEDYEGNPSNFGVGLKRHLSDLAKG"]
encoded = tokenizer(sequences, padding=True, return_tensors="pt").to(device)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# ==== 7. 模型推理 ====
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(logits.logits, dim=-1)

# ==== 8. 输出结果（去除 <cls> 和 <eos>） ====
pred_labels = predictions.tolist()[0][1:-1]
print("Predicted labels:", pred_labels)


hydrophobic_aa = ['V', 'I', 'L', 'A', 'M']
res_type = []
res_ids = []
for i, label in enumerate(pred_labels):
    if label == 1:
        res_ids.append(i + 1)
        res_type.append(sequences[0][i])
print(len(res_ids))
print('sele res ' + '+'.join(map(str, res_ids)))
print('Hydrophobic residues predicted:', res_type)

seq_len = len(sequences[0])
hydrophobic_aa_all_count = sum([1 for aa in sequences[0] if aa in hydrophobic_aa])
print('Sequence length:', seq_len)
print('Hydrophobic AA total count:', hydrophobic_aa_all_count)
print('Predicted hydrophobic count:', len(res_type))
