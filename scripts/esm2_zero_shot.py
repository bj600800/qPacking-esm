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

from official_esm2_zero_shot import remove_insertions, read_msa, compute_pppl


def main(model_path, model_name, sequence, dms_input, offset_idx, mutation_col, scoring_strategy, dms_output):
    df = pd.read_csv(dms_input)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")  # 强制使用MPS
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # shape: [1, L]


    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16).to(device)

    def label_row(row, sequence, token_probs, alphabet, offset_idx):
        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded = tokenizer.convert_tokens_to_ids(wt)
        mt_encoded = tokenizer.convert_tokens_to_ids(mt)

        # add 1 for BOS
        score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        return score.item()

    if scoring_strategy == "wt-marginals":
        with torch.no_grad():
            logits = model(**inputs).logits  # shape: [1, L, vocab_size]
            token_probs = torch.log_softmax(logits, dim=-1)  # probabilities
        df[model_name] = df.apply(
            lambda row: label_row(
                row[mutation_col],
                sequence,
                token_probs,
                tokenizer,
                offset_idx
            ),
            axis=1
        )

    elif scoring_strategy == "masked-marginals":
        all_token_probs = []
        for i in tqdm(range(input_ids.size(1))):
            masked_inputs = input_ids.clone()
            masked_inputs[0, i] = tokenizer.mask_token_id
            with torch.no_grad():
                logits = model(input_ids=masked_inputs).logits
                probs = torch.log_softmax(logits, dim=-1)
            all_token_probs.append(probs[:, i])  # [1, vocab]
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)  # [1, L, vocab]
        df[model_name] = df.apply(
            lambda row: label_row(
                row[mutation_col],
                sequence,
                token_probs,
                tokenizer,
                offset_idx
            ),
            axis=1
        )

    elif scoring_strategy == "pseudo-ppl":
        tqdm.pandas()
        df[model_name] = df.progress_apply(
            lambda row: compute_pppl(
                row[mutation_col],
                sequence,
                model,
                tokenizer,
                offset_idx
            ),
            axis=1
        )


    print("write to file.")
    df.to_csv(dms_output)


if __name__ == "__main__":
    model_path = "/Users/douzhixin/Developer/qPacking/checkpoints/huggingface"
    model_name = "esm2_t33_650M_UR50D"
    sequence = "HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    dms_input = "/Users/douzhixin/Developer/qPacking/data/benchmark/BLAT_ECOLX_Ranganathan2015_labeled.csv"
    offset_idx = 24
    mutation_col = "mutant"
    scoring_strategy = "wt-marginals"
    dms_output = "/Users/douzhixin/Developer/qPacking/data/benchmark/BLAT_ECOLX_Ranganathan2015_hugg_predicted.csv"
    main(model_path, model_name, sequence, dms_input, offset_idx, mutation_col, scoring_strategy, dms_output)