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

def tokenwise_supervised_contrastive_batch_loss(proj_emb, labels, attention_mask=None, temperature=0.07):
    """
    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

    Implements the supervised contrastive loss over a batch of sequences,
    computing loss only within each individual sequence (no cross-sequence comparison).

    The formula implemented is:

        L_supcon_batch = (1 / |B|) * sum_{b=1}^{|B|} [
            (1 / |I^(b)|) * sum_{i∈I^(b)} [
                -(1 / |P(i)|) * sum_{p∈P(i)} log p_{i,p}
            ]
        ]

    where:
        - p_{i,p} = exp(z_i • z_p / τ) / sum_{a∈A(i)} exp(z_i • z_a / τ)
          is the probability of positive sample p given anchor i.

    Notations:
        - B: batch of sequences, |B| is the batch size (number of sequences).
        - I^(b): set of anchor positions i in sequence b (all valid tokens).
        - P(i): set of positive samples p for anchor i (tokens in the same cluster, same sequence).
        - A(i): all valid tokens a in the same sequence as i (excluding itself).
        - z_i, z_p, z_a: normalized embeddings for anchor, positive, and all valid tokens.
        - τ: temperature scaling factor.

    The loss averages over all anchors in each sequence, then averages across all sequences in the batch.

    Returns:
        batch loss: averaged loss over valid sequences in a batch.

    """
    device = proj_emb.device
    batch_size = proj_emb.size(0)
    total_loss = 0.0
    valid_seq_count = 0

    for b in range(batch_size):
        # process each sequence b in the batch

        if attention_mask is not None:  # 0 for padding, 1 for else tokens
            valid_mask = attention_mask[b].bool()
        else:
            valid_mask = torch.ones(proj_emb.size(1), dtype=torch.bool, device=device)  # all true

        if valid_mask.sum() < 2:
            continue

        label_valid_mask = labels[b] != -100
        valid_mask = valid_mask & label_valid_mask

        # valid emb and labels for seq b
        emb = proj_emb[b, valid_mask]  # [N, D]
        lab = labels[b, valid_mask].contiguous().view(-1, 1)  # [N, 1]
        N = emb.shape[0]

        # positive sample mask
        label_mask = torch.eq(lab, lab.T).float()  # [N, N]
        # except self, all sample pairs
        logits_mask = torch.ones_like(label_mask) - torch.eye(N, device=device)
        pos_mask = label_mask * logits_mask

        # similarity logits for each residue pair
        logits = torch.div(torch.matmul(emb, emb.T), temperature)  # [N, N]

        # logits stabilization
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Denominator without sum
        exp_logits = torch.exp(logits) * logits_mask  # [N, N]

        # softmax
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  # [N, N] sum k dim, plus number avoid log0

        sum_anchor_pos_log_prob = (pos_mask * log_prob).sum(1)  # [N]
        # divide by P^{(i)} eliminates the effect of different number of positive samples for each anchor
        anchor_loss= sum_anchor_pos_log_prob / (pos_mask.sum(1) + 1e-12)  # mask.sum(1) is P^{(i)}: number of positive anchors for each i
        # avg loss cross all anchors to get seq loss: sum and divide by I^{(b)}
        loss_seq = -anchor_loss.mean()
        # mean loss across all valid sequences
        total_loss += loss_seq
        valid_seq_count += 1
    loss_batch = total_loss / valid_seq_count if valid_seq_count > 0 else torch.tensor(0.0, device=device)
    return loss_batch.unsqueeze(0)
