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
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
import torch.nn as nn
from torch.nn import MSELoss
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
                                     # weights_only=True,
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
    return loss_batch.unsqueeze(0)

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

class TokenClassificationModel(nn.Module):
    def __init__(self, model_dir, num_class, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.model = load_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_class)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

class TokenRegressionModel(nn.Module):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.model = load_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)  # 每个token一个实数
        self.loss_fn = MSELoss(reduction="mean")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.regressor(hidden_states).squeeze(-1)  # shape: (batch_size, seq_len)

        loss = None
        if labels is not None:
            active_mask = labels != -100
            active_logits = logits[active_mask]
            active_labels = labels[active_mask].float()
            loss = self.loss_fn(active_logits, active_labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits  # shape: (batch_size, seq_len)
        )


class TokenRegressionWeightedModel(nn.Module):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.model = load_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)  # 每个 token 一个实数

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.regressor(hidden_states).squeeze(-1)  # shape: (batch_size, seq_len)

        loss = None
        if labels is not None:
            active_mask = labels != -100
            active_logits = logits[active_mask]
            active_labels = labels[active_mask].float()

            mean_label = active_labels.mean() + 1e-6
            weights = active_labels / mean_label  # focus on the bigger labels

            loss = torch.mean(weights * (active_logits - active_labels) ** 2)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits  # shape: (batch_size, seq_len)
        )


@dataclass
class RegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction: torch.FloatTensor = None


class FitnessRegressionModel(nn.Module):
    def __init__(self, model_dir, model_src, unfreeze_last_n=2):
        super().__init__()
        if model_src == 'official':
            encoder = EsmModel.from_pretrained(model_dir, add_pooling_layer=False)
            model_prefix = "encoder.layer"
        elif model_src == 'finetuned':
            peft_config = PeftConfig.from_pretrained(model_dir)
            model_tuned_encoder = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
            encoder = PeftModel.from_pretrained(model_tuned_encoder, model_dir)
            model_prefix = "base_model.model.encoder.layer"
        else:
            raise ValueError(f"Unsupported model source: {model_src}. Use 'official' or 'finetuned'.")

        self.model = encoder
        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(hidden_size, 1)  # 输出一个实数值
        self.loss_fn = MSELoss(reduction="mean")
        # # 冻结 model，仅训练 regressor
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # 自动检测层数并解冻最后 n 层
        if unfreeze_last_n > 0:
            layer_nums = set()
            for name, _ in self.model.named_parameters():
                if model_prefix in name:
                    try:
                        layer_id = int(name.split(model_prefix + ".")[1].split(".")[0])
                        layer_nums.add(layer_id)
                    except:
                        continue

            if not layer_nums:
                print("❌ 没有找到匹配的 encoder 层名，检查 encoder_prefix 是否正确。")
            else:
                total_layers = max(layer_nums) + 1  # 层数是从 0 开始的
                target_layers = list(range(total_layers - unfreeze_last_n, total_layers))
                print(f"🧠 模型总共 {total_layers} 层，将解冻最后 {unfreeze_last_n} 层: {target_layers}")

                matched_count = 0
                for name, param in self.model.named_parameters():
                    if any(f"{model_prefix}.{i}." in name for i in target_layers):
                        param.requires_grad = True
                        matched_count += 1
                        print(f"✅ Unfroze: {name}")
                    else:
                        param.requires_grad = False

                print(f"\n✅ 解冻完成，共解冻 {matched_count} 个参数项。")

        else:
            # 全冻结
            for _, param in self.model.named_parameters():
                param.requires_grad = False

    def forward(self, wt_input_ids, wt_attention_mask, mut_input_ids, mut_attention_mask, labels=None):
        # 获取 wild-type 和 mutant 的 CLS 表达
        wt_out = self.model(input_ids=wt_input_ids, attention_mask=wt_attention_mask).last_hidden_state[:, 0]
        mut_out = self.model(input_ids=mut_input_ids, attention_mask=mut_attention_mask).last_hidden_state[:, 0]

        # 差向量
        diff = mut_out - wt_out
        out = self.dropout(diff)
        prediction = self.regressor(out).squeeze(-1)  # shape: (B,)

        loss = None
        if labels is not None:
            loss = self.loss_fn(prediction, labels)

        return RegressionOutput(loss=loss, prediction=prediction)

if __name__ == '__main__':
    from transformers import EsmTokenizer, EsmModel
    from peft import PeftModel, PeftConfig
    # --------------------------
    # 加载模型路径（替换为你的路径）
    # --------------------------
    best_model_path = "/Users/douzhixin/Developer/qPacking/code/checkpoints/80/20250710_hydrophobic-binary_esm2-150_80_v1/best"  # 替换为你的 LoRA 模型目录
    peft_config = PeftConfig.from_pretrained(best_model_path)

    # --------------------------
    # 1. 加载未微调模型
    # --------------------------
    model_base = FitnessRegressionModel(best_model_path, 'official')
    model_base.eval()

    # --------------------------
    # 2. 加载微调后的模型（PEFT）
    # --------------------------
    model_tuned = FitnessRegressionModel(best_model_path, 'finetuned')
    model_tuned.eval()

    # --------------------------
    # 准备模拟输入（B=2, L=512）
    # --------------------------
    batch_size = 2
    seq_len = 512
    model_base_encoder = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
    vocab_size = model_base_encoder.config.vocab_size  # ESM vocab size

    dummy_input = {
        'wt_input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'wt_attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long),
        'mut_input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'mut_attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long),
    }

    # --------------------------
    # 验证 forward 传播
    # --------------------------
    with torch.no_grad():
        base_out = model_base(**dummy_input)
        tuned_out = model_tuned(**dummy_input)
        print(f"未微调模型输出: {base_out}")  # torch.Size([2])
        print(f"微调后模型输出: {tuned_out}")  # torch.Size([2])



