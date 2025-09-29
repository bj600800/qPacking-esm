"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/29

# Description: 【更正】models for token-level classification, regression and contrastive learning tasks.
# ------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from .base import BaseESMLoraModel
from .heads import ClassificationHead, RegressionHead, ContrastiveHead, RegressionOutput

class TokenClassificationModel(BaseESMLoraModel):
    def __init__(self, model_dir, num_class, lora_rank, lora_alpha, lora_dropout):
        super().__init__(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.head = ClassificationHead(self.hidden_size, num_class)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encode(input_ids, attention_mask)
        return self.head(hidden, labels)

class TokenRegressionModel(BaseESMLoraModel):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout, weighted=False):
        super().__init__(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.head = RegressionHead(self.hidden_size, weighted=weighted)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encode(input_ids, attention_mask)
        return self.head(hidden, labels)

class HydrophobicContrastiveModel(BaseESMLoraModel):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout, proj_dim, contrastive_loss_fn):
        super().__init__(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.head = ContrastiveHead(self.hidden_size, proj_dim, contrastive_loss_fn)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encode(input_ids, attention_mask)
        return self.head(hidden, labels, attention_mask)