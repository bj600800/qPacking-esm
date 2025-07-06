"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/3

# Description: Continual learning model for token-wise tasks.
# ------------------------------------------------------------------------------
"""
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import EsmModel
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.info(f"{name} 🔥 Trainable")
        else:
            logger.info(f"{name} 🧊 Frozen")
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {round(100 * trainable_params / all_param, 2)}%"
    )


def load_model(model_dir, lora_rank, lora_alpha, lora_dropout):
    """
    load the ESM backbone model with lora
    Args:
        model_dir: denoted as model name

    Returns:
        model
    """
    model = EsmModel.from_pretrained(model_dir,
                                     use_safetensors=False,
                                     torch_dtype=torch.float32,
                                     add_pooling_layer=False)

    # model.gradient_checkpointing_enable()  # reduce the number of stored activations
    model.enable_input_require_grads()  # allow lora update

    config = LoraConfig(
        r=lora_rank,  # attention rank 8
        lora_alpha=lora_alpha,  # alpha scaling 8
        target_modules=[
            "query",
            "key",
            "value",
            "dense"],  # keyword match
        inference_mode=False,
        lora_dropout=lora_dropout, # 0.05
        bias="none"
    )

    model = get_peft_model(model, config)

    # print_trainable_parameters(model)

    return model

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
        device = x.device  # 保证所有计算都在 x 的 device 上

        if self.alpha is not None and self.alpha.device != device:
            self.alpha = self.alpha.to(device)
            self.nll_loss.weight = self.alpha  # 更新 NLLLoss 内部的 weight

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

class HydrophobicBinaryClassificationModel(nn.Module):
    def __init__(self, model_dir, num_clusters, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.model = load_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_clusters)  # binary classification, num_clusters=2

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)  # [batch, seq_len, hidden=1280]
        logits = self.classifier(hidden_states)  # [batch, seq_len, num_clusters] 1280 -> num_clusters

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # reshape to [B*L, C] vs [B*L]
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def save_pretrained(self, save_directory, **kwargs):
        """
        Allow Trainer to call model.save_pretrained() directly.
        Delegates saving to the underlying backbone model.
        """
        self.model.save_pretrained(save_directory, **kwargs)


def tokenwise_supervised_contrastive_loss_batch(proj_emb, labels, attention_mask=None, temperature=0.07):
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
        ##

        # get seq b valid tokens using attention mask
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

        # softmax for all
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  # [N, N] sum k dim, plus number avoid log0
        # sum of positive anchor log probabilities
        sum_anchor_pos_log_prob = (pos_mask * log_prob).sum(1)  # [N]
        # divide by P^{(i)} eliminates the effect of different number of positive samples for each anchor
        anchor_loss= sum_anchor_pos_log_prob / (pos_mask.sum(1) + 1e-12)  # mask.sum(1) is P^{(i)}: number of positive anchors for each i
        # avg loss cross all anchors to get seq loss: sum and divide by I^{(b)}
        loss_seq = -anchor_loss.mean()
        # mean loss across all valid sequences
        total_loss += loss_seq
        valid_seq_count += 1
    loss_batch = total_loss / valid_seq_count if valid_seq_count > 0 else torch.tensor(0.0, device=device)
    return loss_batch


class HydrophobicContrastiveModel(nn.Module):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout, proj_dim):
        super().__init__()
        self.model = load_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(self.model.config.hidden_size, proj_dim)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, L, hidden_size]
        hidden_states = self.dropout(hidden_states)
        proj_emb = self.proj(hidden_states)        # [B, L, proj_dim]
        proj_emb = F.normalize(proj_emb, dim=-1)   # normalized after embedding projection

        loss = None
        if labels is not None:
            loss = tokenwise_supervised_contrastive_loss_batch(proj_emb, labels, attention_mask)

        return TokenClassifierOutput(
            loss=loss,
            logits=proj_emb,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def save_pretrained(self, save_directory, **kwargs):
        """
        Allow Trainer to call model.save_pretrained() directly.
        Delegates saving to the underlying backbone model.
        """
        self.model.save_pretrained(save_directory, **kwargs)


if __name__ == '__main__':
    from transformers import EsmTokenizer
    import torch

    model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t33_650M_UR50D"
    # 用ESM对应的蛋白质字母表初始化tokenizer（模拟真实序列用）
    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)

    # 模拟2条蛋白质序列
    sequences = [
        "MKT",
        "VLS"
    ]

    # 将序列编码成input_ids、attention_mask
    encoded = tokenizer(sequences, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"]  # [B, L]
    print('input_ids: ', input_ids)
    attention_mask = encoded["attention_mask"]  # [B, L]
    print('attention_mask: ', attention_mask.shape)
    # 为每个token生成随机簇标签 (0表示非簇 ，1~2为不同簇)
    labels = torch.randint(0, 3, input_ids.shape)  # [B, L]
    labels[0, 0], labels[0, -1], labels[1, 0], labels[-1, -1] = [-100] * 4
    print('labels: ', labels)
    # ====== 创建模型 ======
    model = HydrophobicContrastiveModel(
        model_dir=model_dir,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.05,
        proj_dim=64,
    )

    # ====== 前向传播 ======
    model.train()
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    print("Loss:", output.loss.item())

    # ====== 检查能否正常反向传播 ======
    output.loss.backward()
    print("Backward pass successful!")


    # Test HydrophobicBinaryClassificationModel
    # model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t33_650M_UR50D"
    #
    # # configs params
    # num_clusters = 5
    # lora_rank = 8
    # lora_alpha = 8
    # lora_dropout = 0.05
    #
    # # 加载模型
    # model = HydrophobicBinaryClassificationModel(model_dir, num_clusters, lora_rank, lora_alpha, lora_dropout)
    # print_trainable_parameters(model)
    # model.eval()
    #
    # # 加载 tokenizer（对应 ESM 模型）
    # tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    # # 真实氨基酸序列（可改为你自己的）
    # sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYG"
    #
    # # 使用 tokenizer 编码
    # encoded = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #
    # # 获取 input_ids 和 attention_mask
    # input_ids = encoded["input_ids"]
    # attention_mask = encoded["attention_mask"]  # shape: [1, seq_len]
    #
    # # 推理
    # seq_len = len(sequence)  # 39
    # batch_size = 1
    # num_classes = 5
    #
    # # 生成一个 labels，值在 [0, 4] 之间
    # labels = torch.randint(low=0, high=num_classes, size=(batch_size, seq_len))
    # labels[0, 0] = -100  # 设置第一个位置为 -100，表示忽略该位置的损失计算
    #
    # with torch.no_grad():
    #     logits = torch.randn(batch_size, seq_len, num_classes, requires_grad=True)
    # print(logits.shape) # [1, 37, 5]



