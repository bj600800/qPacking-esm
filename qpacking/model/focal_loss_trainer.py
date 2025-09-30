"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/30

# Description: 
# ------------------------------------------------------------------------------
"""
from transformers import Trainer
import torch
from qpacking.model.loss import FocalLoss

class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_gamma=2.0, focal_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)

        if focal_alpha is not None and not isinstance(focal_alpha, torch.Tensor):
            focal_alpha = torch.tensor(focal_alpha, dtype=torch.float32)

        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
            ignore_index=-100
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        logits = logits.to(device)
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
