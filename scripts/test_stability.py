"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/7/24

# Description: Zero-shot prediction using PEFT-finetuned ESM-2 model with official MLM head,
# adapted to predict fitness from mutation data with separate wt/pos/mt columns.
# ------------------------------------------------------------------------------
"""

import os
from tqdm import tqdm
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from peft import PeftConfig, PeftModel
from qpacking.utils import logger
from scipy.stats import spearmanr

logger = logger.setup_log(name=__name__)


def load_model_and_tokenizer(model_path, device):
    # # 1. 读取PEFT配置，获得基础模型路径
    # peft_config = PeftConfig.from_pretrained(model_path)
    #
    # # 2. 加载官方ESM2基础模型（带MLM头）
    # base_model = EsmForMaskedLM.from_pretrained(peft_config.base_model_name_or_path).to(device)
    #
    # # 3. 加载LoRA微调权重（只针对编码器部分）
    # lora_model = PeftModel.from_pretrained(base_model.esm, model_path).to(device)
    #
    # # 4. 用LoRA模型替换基础模型的编码器
    # base_model.esm = lora_model
    # base_model.eval()
    #
    # # 5. 加载tokenizer（从微调目录）
    # tokenizer = EsmTokenizer.from_pretrained(model_path)
    base_model = EsmForMaskedLM.from_pretrained(model_path).to(device)
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    base_model.eval()
    return tokenizer, base_model


def get_token_probs_wt_marginals(model, inputs):
    with torch.no_grad():
        logits = model(**inputs).logits
        token_probs = torch.log_softmax(logits, dim=-1)
    return token_probs


def get_token_probs_masked_marginals(model, tokenizer, input_ids):
    all_token_probs = []
    for i in tqdm(range(input_ids.size(1))):
        masked_inputs = input_ids.clone()
        masked_inputs[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=masked_inputs).logits
            probs = torch.log_softmax(logits, dim=-1)
        all_token_probs.append(probs[:, i])
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    return token_probs


def label_mutation_score(mutation, sequence, token_probs, tokenizer, offset_idx=29):
    wt, pos, mt = mutation[0], int(mutation[1:-1]), mutation[-1]
    idx = pos - offset_idx  # 结构编号 → 序列索引
    if sequence[idx] != wt:
        logger.warning(f"[SKIP] Wildtype mismatch at position {pos}: expected {wt}, found {sequence[idx]}")
        return float('nan')

    wt_id = tokenizer.convert_tokens_to_ids(wt)
    mt_id = tokenizer.convert_tokens_to_ids(mt)
    score = token_probs[0, 1 + idx, mt_id] - token_probs[0, 1 + idx, wt_id]
    return score.item()


def score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name):
    df[model_name] = df[mutation_col].apply(
        lambda mutation: label_mutation_score(mutation, sequence, token_probs, tokenizer, offset_idx)
    )
    return df


def main(model_path, model_name, sequence, dms_input, offset_idx, scoring_strategy, dms_output):
    df = pd.read_csv(dms_input)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 构造统一突变字符串列 eg: N44A
    df["mutation"] = df.apply(lambda row: f"{row['wtaa']}{row['pos']}{row['mutaa']}", axis=1)
    mutation_col = "mutation"

    tokenizer, model = load_model_and_tokenizer(model_path, device)
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    if scoring_strategy == "wt-marginals":
        token_probs = get_token_probs_wt_marginals(model, inputs)
        df = score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name)

        # 计算Spearman相关性，评估预测和真实fitness关系
        if "fitness" in df.columns:
            corr, pval = spearmanr(df["fitness"].dropna(), df[model_name].dropna())
            print(f"Spearman correlation: {corr:.4f} (p={pval:.2e})")

    elif scoring_strategy == "masked-marginals":
        token_probs = get_token_probs_masked_marginals(model, tokenizer, input_ids)
        df = score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name)

    else:
        raise ValueError("Only wt-marginals and masked-marginals are supported with PEFT models.")

    logger.info(f"Writing to file: {dms_output}")
    df.to_csv(dms_output, index=False)


if __name__ == "__main__":
    model_path = "/Users/douzhixin/Developer/qPacking/code/checkpoints/esm2_t30_150M_UR50D"
    model_name = os.path.dirname(model_path).split("/")[-1]
    sequence = "ERVKIIAEFKKASPSAGDINADASLEDFIRMYDELADAISILTEKHYFKGDPAFVRAARNLTSRPILAKDFYIDTVQVKLASSVGADAILIIARILTAEQIKEIYEAAEELGMDSLVEVHSREDLEKVFSVIRPKIIGINTRDLDTFEIKKNVLWELLPLVPDDTVVVAESGIKDPRELKDLRGKVNAVLVGTSIMKAENPRRFLEEMRAWSE"
    dms_input = "/Users/douzhixin/Developer/qPacking/data/benchmark/tm.csv"
    offset_idx = 40
    scoring_strategy = "wt-marginals"  # or "masked-marginals"
    dms_output = "/Users/douzhixin/Developer/qPacking/data/benchmark/tm_predicted.csv"

    main(model_path, model_name, sequence, dms_input, offset_idx, scoring_strategy, dms_output)
