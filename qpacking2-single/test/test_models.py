"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/30

# Description: 
# ------------------------------------------------------------------------------
"""
import torch
import pytest
from qpacking.model.models import TokenClassificationModel


def test_token_classification_model():
    model = TokenClassificationModel(
        model_dir = "/Users/douzhixin/Developer/qPacking/data/checkpoints/80/20250710_hydrophobic-binary_esm2-150_80_v1/best",
        num_class = 2,
        lora_rank = 8,
        lora_alpha = 8,
        lora_dropout = 0.05
    )
    model.eval()
    batch_size = 2
    seq_len = 20
    vocab_size = model.backbone.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.forward(input_ids, attention_mask)

    assert output.logits.shape == (batch_size, seq_len, 2)