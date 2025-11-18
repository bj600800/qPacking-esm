"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/29

# Description: Base model with LoRA adaptation
[TEST PASS 100%]
# ------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import EsmModel

def load_lora_model(model_dir, lora_rank, lora_alpha, lora_dropout):
    model = EsmModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        add_pooling_layer=False
    )
    model.enable_input_require_grads() # Enable gradients for input embeddings
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query","key","value","dense"],
        lora_dropout=lora_dropout,
        inference_mode=False,
        bias="none"
    )
    return get_peft_model(model, config)


class BaseESMLoraModel(nn.Module):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.backbone = load_lora_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)

    def encode(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)