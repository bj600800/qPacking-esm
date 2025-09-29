"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/29

# Description: Test base ESM LoRA model using pytest
# ------------------------------------------------------------------------------
"""
import torch
import pytest
from qpacking.model.base import BaseESMLoraModel


@pytest.mark.basic
def test_base_esmlora_forward_real_model():
    model_path = "/Users/douzhixin/Developer/qPacking/data/checkpoints/esm2_t30_150M_UR50D"

    model = BaseESMLoraModel(
        model_dir=model_path,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.05
    )
    model.eval()

    batch_size = 2
    seq_len = 20
    vocab_size = model.backbone.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.encode(input_ids, attention_mask)

    # TESTS
    assert output.shape[0] == batch_size
    assert output.shape[1] == seq_len
    assert output.shape[2] == model.hidden_size

    print("Output shape:", output.shape)
