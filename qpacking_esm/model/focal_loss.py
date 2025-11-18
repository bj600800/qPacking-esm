"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/29

# Description: Customized training loss functions
# ------------------------------------------------------------------------------
"""
import torch.nn as nn
from typing import Optional

import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: https://arxiv.org/pdf/1708.02002
    """
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(weight=self.alpha,
                                   reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = x.device

        if self.alpha is not None and self.alpha.device != device:
            self.alpha = self.alpha.to(device)
            self.nll_loss.weight = self.alpha

        if x.ndim > 2:
            batch_size, seq_len, num_classes = x.shape
            x = x.view(-1, num_classes)  # (B * L, C)
            y = y.view(-1)               # (B * L,)

        y = y.to(device)

        mask = y != self.ignore_index
        x, y = x[mask], y[mask]
        if len(y) == 0:
            return torch.tensor(0.0, device=device)

        log_p = F.log_softmax(x, dim=-1)
        ce_loss = self.nll_loss(log_p, y)

        log_pt = log_p[torch.arange(len(y), device=device), y]
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
