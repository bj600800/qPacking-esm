"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/3

# Description: 
# ------------------------------------------------------------------------------
"""
from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    EsmModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import torch.nn as nn
import numpy as np

def load_esm_model(model_dir):
    model = EsmModel.from_pretrained(model_dir, torch_dtype=torch.float32)
    for param in model.parameters():
        param.requires_grad = False

    model.gradient_checkpointing_enable()  # reduce the number of stored activations
    model.enable_input_require_grads()  # allow lora update
    # for name, module in model.named_modules():
    #     print(name)
    # input()

    return model

def load_lora(model):
    config = LoraConfig(
        r=8,  # attention heads
        lora_alpha=8,  # alpha scaling
        target_modules=[
            "query",
            "key",
            "value",
            "dense"],  # keyword match
        inference_mode=False,
        lora_dropout=0.05,
        bias="none"
    )

    # Allow the parameters of the last transformer block to be updated during fine-tuning
    for param in model.encoder.layer[-1:].parameters():
        param.requires_grad = True

    model = get_peft_model(model, config)

    print_trainable_parameters(model)

    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}%"
    )

class EsmRegressionModel(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.model = load_esm_model(model_dir)
        self.model = load_lora(self.model)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        hidden_states = self.dropout(hidden_states)
        predictions = self.regressor(hidden_states).squeeze(-1)  # [batch, seq_len]
        return predictions


if __name__ == '__main__':
    import torch
    model_dir = r"/Users/douzhixin/Developer/qPacking/checkpoints/esm2_t33_650M_UR50D"


