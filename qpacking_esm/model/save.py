"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/30

# Description: 
# ------------------------------------------------------------------------------
"""
import os
import torch
from transformers import TrainerCallback

class SaveCompleteModelCallback(TrainerCallback):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = self.model

        # save backbone with lora. Need to notice, HF automated save safetensors file

        if hasattr(model, "backbone"):
            model.backbone.save_pretrained(checkpoint_dir)

        # save header
        if hasattr(model, "head"):
            torch.save(model.head.state_dict(), os.path.join(checkpoint_dir, "task_head.pt"))

        # save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
