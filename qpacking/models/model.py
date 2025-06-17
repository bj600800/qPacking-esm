"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/3

# Description: Continual learning model for token-wise tasks.
# ------------------------------------------------------------------------------
"""
import torch
from transformers import EsmModel
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import torch.nn as nn


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
            print(f"{name}")
        else:
            print(f"{name} 🧊 Frozen")
    print(
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


class TokenClassificationModel(nn.Module):
    def __init__(self, model_dir, num_clusters, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        self.model = load_model(model_dir, lora_rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_clusters)  # nn.Linear(input, output)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)  # [batch, seq_len, hidden=1280]
        logits = self.classifier(hidden_states)  # [batch, seq_len, num_clusters] 1280 -> num_clusters

        return logits


if __name__ == '__main__':
    from transformers import EsmTokenizer
    model_dir = r"/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t33_650M_UR50D"

    # config params
    num_clusters = 5
    lora_rank = 8
    lora_alpha = 8
    lora_dropout = 0.05

    # 加载模型
    model = TokenClassificationModel(model_dir, num_clusters, lora_rank, lora_alpha, lora_dropout)
    model.eval()

    # 加载 tokenizer（对应 ESM 模型）
    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    # 真实氨基酸序列（可改为你自己的）
    sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYG"

    # 使用 tokenizer 编码
    encoded = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 获取 input_ids 和 attention_mask
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]  # shape: [1, seq_len]

    # 推理
    seq_len = len(sequence)  # 39
    batch_size = 1
    num_classes = 5

    # 生成一个 labels，值在 [0, 4] 之间
    labels = torch.randint(low=0, high=num_classes, size=(batch_size, seq_len))
    labels[0, 0] = -100  # 设置第一个位置为 -100，表示忽略该位置的损失计算

    with torch.no_grad():
        logits = torch.randn(batch_size, seq_len, num_classes, requires_grad=True)
    print(logits.shape) # [1, 37, 5]



