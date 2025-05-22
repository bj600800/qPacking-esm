"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/5/21

# Description: Zero-shot prediction with esm2-650M adapted part of the official codes from esm github.
# ------------------------------------------------------------------------------
"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import official
device = torch.device("mps" if torch.mps.is_available() else "cpu")  # 强制使用MPS

model_path = "/Users/douzhixin/Developer/qPacking/checkpoints/huggingface"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16).to(device)

# 验证设备位置
print(f"Model device: {next(model.parameters()).device}")  # 应输出 mps:0


# 推理示例
def run_inference(text="MKTIIALSYIFCLVFA"):
    inputs = tokenizer(
        text,
        return_tensors="pt", # different options [numpy：np, pytorch：pt, tensorflow：tf]
        padding=True,
        max_length=512,
        truncation=True
    ).to(device)

    with torch.no_grad(), torch.autocast(device_type='mps'):  # 自动混合精度
        outputs = model(**inputs)

    return torch.softmax(outputs.logits, dim=-1)


# 测试运行
if __name__ == "__main__":
    try:
        probs = run_inference()
        print("MPS inference successful!")
        print(f"Output shape: {probs.shape}")
        print(probs)
        
    except RuntimeError as e:
        print(f"MPS Error: {e}\n建议：")
        print("1. 升级PyTorch到最新版 (>=2.3)")
        print("2. 检查模型文件完整性")
        print("3. 添加环境变量: export PYTORCH_ENABLE_MPS_FALLBACK=1")