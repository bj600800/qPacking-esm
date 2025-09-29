"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/3

# Description: qPacking Model definition， the original code
# ------------------------------------------------------------------------------
"""
from typing import Optional
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
from torch.nn import MSELoss

from qpacking.model.loss import tokenwise_supervised_contrastive_batch_loss
from qpacking.model import params
from qpacking.common import logger
logger = logger.setup_log(name=__name__)


# def load_model(model_dir, lora_rank, lora_alpha, lora_dropout):
#     """
#     load the ESM backbone model with lora
#     Args:
#         model_dir: denoted as model name
#
#     Returns:
#         model
#     """
#     model = EsmModel.from_pretrained(model_dir,
#                                      # weights_only=True,
#                                      torch_dtype=torch.float32,
#                                      add_pooling_layer=False)
#     # model.gradient_checkpointing_enable()  # reduce the number of stored activations
#     model.enable_input_require_grads()  # allow lora update
#
#     config = LoraConfig(
#         r=lora_rank,  # attention rank 8
#         lora_alpha=lora_alpha,  # alpha scaling 8
#         target_modules=[
#             "query",
#             "key",
#             "value",
#             "dense"],  # keyword match
#         inference_mode=False,
#         lora_dropout=lora_dropout, # 0.05
#         bias="none"
#     )
#
#     model = get_peft_model(model, config)
#
#     return model


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
            loss = tokenwise_supervised_contrastive_batch_loss(proj_emb, labels, attention_mask)
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
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)
        self.loss_fn = MSELoss(reduction="mean")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.regressor(hidden_states).squeeze(-1)

        loss = None
        if labels is not None:
            active_mask = labels != -100
            active_logits = logits[active_mask]
            active_labels = labels[active_mask].float()
            loss = self.loss_fn(active_logits, active_labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )


class TokenRegressionWeightedModel(nn.Module):
    def __init__(self, model_dir, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.model = load_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.regressor(hidden_states).squeeze(-1)

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
    def __init__(self, model_dir, model_src, unfreeze_last_n, emb_src):
        super().__init__()
        if model_src == 'official':
            self._keys_to_ignore_on_save = ['regressor', 'loss_fn']
            encoder = EsmModel.from_pretrained(model_dir, add_pooling_layer=False)
            model_prefix = "encoder.layer"
        elif model_src == 'finetuned':
            peft_config = PeftConfig.from_pretrained(model_dir)
            model_tuned_encoder = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
            encoder = PeftModel.from_pretrained(model_tuned_encoder, model_dir)
            model_prefix = "base_model.model.encoder.layer"
        else:
            raise ValueError(f"Unsupported model source: {model_src}. Use 'official' or 'finetuned'.")
        self.emb_src = emb_src
        self.model = encoder
        hidden_size = self.model.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # self.regressor = nn.Linear(hidden_size, 1)  # the simplest linear regression head

        self.loss_fn = MSELoss(reduction="mean")

        # unfrozen model params
        params.unfreeze_backbone(self.model, unfreeze_last_n, model_prefix)

        # Train header
        for name, param in self.regressor.named_parameters():
            param.requires_grad = True


    def forward(self, wt_input_ids, wt_attention_mask, mut_input_ids, mut_attention_mask, mutation_pos, labels):
        if self.emb_src == 'cls':
            wt_out = self.model(wt_input_ids, wt_attention_mask)[0][:, 0]  # CLS token embedding
            mut_out = self.model(mut_input_ids, mut_attention_mask)[0][:, 0]

        elif self.emb_src == 'pos':
            wt_hidden = self.model(wt_input_ids, wt_attention_mask)[0]  # (B, L, H)
            mut_hidden = self.model(mut_input_ids, mut_attention_mask)[0]

            B, L, H = wt_hidden.size()
            pos = mutation_pos.view(-1, 1, 1).expand(-1, 1, H)  # (B,1,H)
            wt_out = torch.gather(wt_hidden, 1, pos).squeeze(1)  # (B, H)
            mut_out = torch.gather(mut_hidden, 1, pos).squeeze(1)

        else:
            raise ValueError(f"Unsupported emb_src: {self.emb_src}")

        diff = mut_out - wt_out

        prediction = self.regressor(diff).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(prediction, labels)

        return RegressionOutput(loss=loss, prediction=prediction)


if __name__ == '__main__':
    from transformers import EsmModel
    from peft import PeftModel, PeftConfig
    best_model_path = "/Users/douzhixin/Developer/qPacking/data/checkpoints/80/20250710_hydrophobic-binary_esm2-150_80_v1/best"
    peft_config = PeftConfig.from_pretrained(best_model_path)
    model_base = FitnessRegressionModel(best_model_path, 'official', unfreeze_last_n=0, emb_src='cls')
    model_base.eval()
    model_tuned = FitnessRegressionModel(best_model_path, 'finetuned', unfreeze_last_n=0, emb_src='cls')
    model_tuned.eval()
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
    with torch.no_grad():
        base_out = model_base(**dummy_input)
        tuned_out = model_tuned(**dummy_input)
        print(f"Before finetuning: {base_out}")  # torch.Size([2])
        print(f"After finetuning: {tuned_out}")  # torch.Size([2])



