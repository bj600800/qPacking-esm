"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/5/21

# Description: Zero-shot prediction with esm2-650M adapted part of the official codes from esm github.
# ------------------------------------------------------------------------------
"""
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from official_esm2_zero_shot import compute_pppl


def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    return tokenizer, model


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


def label_mutation_score(mutation, sequence, token_probs, tokenizer, offset_idx):
    wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
    assert sequence[idx] == wt, f"Wildtype {wt} does not match sequence at position {idx}: {sequence[idx]}"
    wt_encoded = tokenizer.convert_tokens_to_ids(wt)
    mt_encoded = tokenizer.convert_tokens_to_ids(mt)
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()


def score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name):
    df[model_name] = df[mutation_col].apply(
        lambda mutation: label_mutation_score(mutation, sequence, token_probs, tokenizer, offset_idx)
    )
    return df


def score_with_pseudo_ppl(df, sequence, model, tokenizer, offset_idx, mutation_col, model_name):
    tqdm.pandas()
    df[model_name] = df[mutation_col].progress_apply(
        lambda mutation: compute_pppl(mutation, sequence, model, tokenizer, offset_idx)
    )
    return df


def main(model_path, model_name, sequence, dms_input, offset_idx, mutation_col, scoring_strategy, dms_output):
    df = pd.read_csv(dms_input)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer, model = load_model_and_tokenizer(model_path, device)
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    if scoring_strategy == "wt-marginals":
        token_probs = get_token_probs_wt_marginals(model, inputs)
        df = score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name)

    elif scoring_strategy == "masked-marginals":
        token_probs = get_token_probs_masked_marginals(model, tokenizer, input_ids)
        df = score_mutations(df, sequence, token_probs, tokenizer, offset_idx, mutation_col, model_name)

    elif scoring_strategy == "pseudo-ppl":
        df = score_with_pseudo_ppl(df, sequence, model, tokenizer, offset_idx, mutation_col, model_name)

    print("Writing to file:", dms_output)
    df.to_csv(dms_output, index=False)


if __name__ == "__main__":
    model_path = "/Users/douzhixin/Developer/qPacking/checkpoints/huggingface"
    model_name = "esm2_t33_650M_UR50D"
    sequence = "HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    dms_input = "/Users/douzhixin/Developer/qPacking/data/benchmark/BLAT_ECOLX_Ranganathan2015_labeled.csv"
    offset_idx = 24
    mutation_col = "mutant"
    scoring_strategy = "wt-marginals"  # or "masked-marginals" or "pseudo-ppl"
    dms_output = "/Users/douzhixin/Developer/qPacking/data/benchmark/BLAT_ECOLX_Ranganathan2015_hugg_predicted.csv"
    main(model_path, model_name, sequence, dms_input, offset_idx, mutation_col, scoring_strategy, dms_output)
