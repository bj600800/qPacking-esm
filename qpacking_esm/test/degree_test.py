"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/15

# Description: Test fine-tuned ESM-2 model for per-residue degree regression
# ------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel
from peft import PeftModel, PeftConfig

best_model_path = "/Users/douzhixin/Developer/qPacking-esm/data/test/checkpoints/degree/best"
tokenizer = EsmTokenizer.from_pretrained(best_model_path)
peft_config = PeftConfig.from_pretrained(best_model_path)
base_model = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
backbone = PeftModel.from_pretrained(base_model, best_model_path)

class HydrophobicDegreeRegressionModel(nn.Module):
    def __init__(self, backbone, regressor_path):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(regressor_path, weights_only=True))

    def forward(self, input_ids, attention_mask=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden).squeeze(-1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = HydrophobicDegreeRegressionModel(backbone, f"{best_model_path}/regression_head.pt").to(device)
model.eval()

sequences = ["IRGVNLGSLFVFEPWIANNEWNTMGCGGQQSEFDCVMNTGQERSDAAFQKHWDTWITEGDLDEMMSYGINTIRIPVGYWLDESLVDQNSEHFPRGAVKYLIRLCGWASDRGFYIILDQHGAPGAQVAKNSFTGQFANTPGFYNDYQYGRAVKFLEFLRKLAHDNNELRNVGTIELVNEPTNWDSSVQSLRSTFYKNGYNAIRNVEKSLGVSANNYFHIQMMSSLWGSGNPTEFLDDTYFTAFDDHRYLKWANKNDVPWTHDSYISTSCNDNRNGDASGPTIVGEWSISPPDEIENSDDWNRDTQKDFYKKWFAAQVHAYEKNTAGWVFWTWKAQLGDYRWSYRDGVIAGVIPRDLNSIASS"]

encoded = tokenizer(sequences, padding=True, return_tensors="pt")
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

with torch.no_grad():
    preds = model(input_ids=input_ids, attention_mask=attention_mask)  # shape [1, L]
    degree_scores = preds[0][1:-1].tolist()  # 去掉 [CLS] 和 [SEP]

hydrophobic_aa = {'V', 'I', 'L', 'A', 'M'}
masked_results = []

for i, (aa, score) in enumerate(zip(sequences[0], degree_scores)):
    if i < 10:
        continue
    if aa in hydrophobic_aa:
        masked_results.append((i + 1, aa, score))  # 位置从1开始计数


print(f"\nFiltered Hydrophobic Degrees (after skipping first 10 residues): {len(masked_results)} residues")
for pos, aa, score in masked_results:
    print(f"Residue {pos} ({aa}): Degree = {score:.4f}")

if masked_results:
    values = [x[2] for x in masked_results]
    print(f"\nMean Degree of Selected Hydrophobic Residues: {sum(values) / len(values):.4f}")
else:
    print("\nNo hydrophobic residues found after position 10.")
