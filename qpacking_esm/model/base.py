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
    num_layers = len(model.encoder.layer)
    # 只在最后3层添加LoRA
    target_modules = []
    for i in range(max(0, num_layers - 1), num_layers):
        layer_prefix = f"encoder.layer.{i}."
        target_modules.extend([
            f"{layer_prefix}attention.self.query",
            f"{layer_prefix}attention.self.key",
            f"{layer_prefix}attention.self.value",
            f"{layer_prefix}attention.output.dense",
            f"{layer_prefix}output.dense"
        ])

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        inference_mode=False,
        bias="none"
    )

    return get_peft_model(model, config)


class BaseESMLoraModel(nn.Module):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.backbone = load_lora_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        # 方法1：打印LoRA层
        print("LoRA Layers:")
        lora_count = 0
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                print(f"   {name}")
                lora_count += 1
        print(f"\n2. Total LoRA layers found: {lora_count}")

        # 方法3：参数统计
        print("\n3. Parameter Statistics:")
        self.backbone.print_trainable_parameters()
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)

    def encode(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)