"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/6/3

# Description: Continual learning model for token-wise tasks.
# ------------------------------------------------------------------------------
"""
from transformers import (
    EsmTokenizer,
    EsmModel,
    EsmConfig)
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


def load_model(model_dir):
    """
    load the ESM backbone model with lora
    Args:
        model_dir: denoted as model name

    Returns:
        model
    """
    model = EsmModel.from_pretrained(model_dir,
                                     torch_dtype=torch.float32,
                                     add_pooling_layer=False)

    model.gradient_checkpointing_enable()  # reduce the number of stored activations
    model.enable_input_require_grads()  # allow lora update

    config = LoraConfig(
        r=8,  # attention heads
        lora_alpha=8,  # alpha scaling
        target_modules=[
            "query",
            "key",
            "value",
            "dense"],  # keyword match
        inference_mode=False,
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, config)

    print_trainable_parameters(model)

    return model


class TokenClassificationModel(nn.Module):
    def __init__(self, model_dir, num_clusters):
        super().__init__()
        self.model = load_model(model_dir)
        print(self.model.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_clusters)  # nn.Linear(input, output)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)  # [batch, seq_len, hidden=1280]
        class_logits = self.classifier(hidden_states)  # [batch, seq_len, num_clusters] 1280 -> num_clusters
        return class_logits


if __name__ == '__main__':
    import torch
    model_dir = r"/Users/douzhixin/Developer/qPacking/checkpoints/esm2_t33_650M_UR50D"
    num_clusters = 5

    # 加载模型
    model = TokenClassificationModel(model_dir, num_clusters)
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
    print(attention_mask)

    # 推理
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    print("Input shape:", input_ids.shape)
    print("Output logits shape:", logits.shape)  # [1, seq_len, num_clusters]



