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
    """
    saved 1
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = args.output_dir
        output_dir = os.path.join(checkpoint_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)

        # save adapter
        if hasattr(self.model, "model") and hasattr(self.model.model, "save_pretrained"):
            self.model.model.save_pretrained(output_dir)

        if hasattr(self.model, "classifier"):
            torch.save(self.model.classifier.state_dict(), os.path.join(output_dir, "classifier_head.pt"))

        if hasattr(self.model, "regressor"):
            torch.save(self.model.regressor.state_dict(), os.path.join(output_dir, "regression_head.pt"))

        self.tokenizer.save_pretrained(output_dir)
