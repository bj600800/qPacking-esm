"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/29

# Description: Models for token-level classification, regression and contrastive learning tasks.
# ------------------------------------------------------------------------------
"""
from qpacking_esm.model.base import BaseESMLoraModel
from qpacking_esm.model.heads import ClassificationHead, RegressionHead

class TokenClassificationModel(BaseESMLoraModel):
    def __init__(self, model_dir, num_class, add_lora_layers, lora_rank, lora_alpha, lora_dropout):
        super().__init__(model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout)
        self.head = ClassificationHead(self.hidden_size, num_class)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encode(input_ids, attention_mask)
        return self.head(hidden, labels)

class TokenRegressionModel(BaseESMLoraModel):
    def __init__(self, model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout, weighted=False):
        super().__init__(model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout)
        self.head = RegressionHead(self.hidden_size, weighted=weighted)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encode(input_ids, attention_mask)
        return self.head(hidden, labels)