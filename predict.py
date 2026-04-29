"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2026/4/29

# Description: Protein mutational effect prediction model.
# ------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import EsmModel, AutoTokenizer
from peft import PeftConfig, PeftModel
from dataclasses import dataclass
from qpacking_esm.common import logger

logger = logger.setup_log(name=__name__)

# =====================Data config========================
HYDROPHOBIC_AA = ["A", "M", "V", "I", "L"]
# =====================Data config========================

# =====================Args config========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--reg_ckpt", type=str, required=True)
parser.add_argument("--fasta_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--positions", type=str, required=True)
args = parser.parse_args()
# =====================Args config========================



def read_seq(fasta_file):
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    return "".join([l.strip() for l in lines if not l.startswith(">")])


class FitnessRegressionModel(nn.Module):
    def __init__(
        self,
        model_dir,
        model_src,
        emb_src,
        reg_ckpt
    ):
        super().__init__()

        if model_src == "official":
            encoder = EsmModel.from_pretrained(
                model_dir,
                add_pooling_layer=False
            )

        elif model_src == "finetuned":
            peft_config = PeftConfig.from_pretrained(model_dir)
            base = EsmModel.from_pretrained(
                peft_config.base_model_name_or_path,
                add_pooling_layer=False
            )
            encoder = PeftModel.from_pretrained(base, model_dir)

        else:
            raise ValueError("model_src must be official or finetuned")

        self.model = encoder
        self.emb_src = emb_src

        hidden = self.model.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )

        if reg_ckpt is not None:
            ckpt = torch.load(reg_ckpt, map_location="cpu")

            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            new_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith("regressor."):
                    new_ckpt[k.replace("regressor.", "")] = v
                else:
                    new_ckpt[k] = v

            self.regressor.load_state_dict(new_ckpt, strict=False)

    # embedding extraction
    def extract(self, input_ids, attention_mask, mutation_pos):

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        if self.emb_src == "cls":
            return out[:, 0]

        elif self.emb_src == "pos":
            B, L, H = out.shape
            pos = mutation_pos.view(-1, 1, 1).expand(-1, 1, H)
            return torch.gather(out, 1, pos).squeeze(1)

        else:
            raise ValueError("emb_src must be cls or pos")

    def forward(
        self,
        wt_input_ids,
        wt_attention_mask,
        mut_input_ids,
        mut_attention_mask,
        mutation_pos
    ):

        wt_emb = self.extract(wt_input_ids, wt_attention_mask, mutation_pos)
        mut_emb = self.extract(mut_input_ids, mut_attention_mask, mutation_pos)

        diff = mut_emb - wt_emb

        pred = self.regressor(diff).squeeze(-1)

        return RegressionOutput(prediction=pred)

@dataclass
class RegressionOutput:
    prediction: torch.Tensor


def load_model(model_path, reg_ckpt, device):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = FitnessRegressionModel(
        model_dir=model_path,
        model_src="finetuned",
        emb_src="pos",
        reg_ckpt=reg_ckpt
    ).to(device)

    model.eval()

    return tokenizer, model

def predict_site(sequence, pos, model, tokenizer, device):

    pos0 = pos - 1
    wt_aa = sequence[pos0]

    wt_inputs = tokenizer(sequence, return_tensors="pt").to(device)

    results = []

    for aa in HYDROPHOBIC_AA:

        if aa == wt_aa:
            continue

        mut_seq = sequence[:pos0] + aa + sequence[pos0 + 1:]
        mut_inputs = tokenizer(mut_seq, return_tensors="pt").to(device)

        mutation_pos = torch.tensor([pos0]).to(device)

        with torch.no_grad():
            output = model(
                wt_input_ids=wt_inputs["input_ids"],
                wt_attention_mask=wt_inputs["attention_mask"],
                mut_input_ids=mut_inputs["input_ids"],
                mut_attention_mask=mut_inputs["attention_mask"],
                mutation_pos=mutation_pos
            )

        results.append({
            "position": pos,
            "wt": wt_aa,
            "mut": aa,
            "fitness": output.prediction.item()
        })

    return results

def scan(sequence, positions, model, tokenizer, device):

    all_results = []

    for p in tqdm(positions):
        all_results.extend(
            predict_site(sequence, p, model, tokenizer, device)
        )

    return pd.DataFrame(all_results)



def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("=" * 50)
    logger.info("MODEL CONFIG")
    logger.info("=" * 50)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Regression ckpt: {args.reg_ckpt}")
    logger.info(f"Device: {device}")

    positions = [int(x) for x in args.positions.split(",")]
    sequence = read_seq(args.fasta_file)

    logger.info("=" * 50)
    logger.info("INPUT SEQUENCE")
    logger.info("=" * 50)
    logger.info(f"Length: {len(sequence)}")
    logger.info(f"Sequence: {sequence}")

    logger.info("=" * 50)
    logger.info("MUTATION INFO")
    logger.info("=" * 50)
    logger.info(f"Target positions: {positions}")
    logger.info(f"Total positions : {len(positions)}")
    logger.info(f"Mutation AA set: {HYDROPHOBIC_AA}")
    logger.info(f"Total mutants: {len(positions) * (len(HYDROPHOBIC_AA) - 1)}")

    logger.info("=" * 50)
    logger.info("MODEL LOADING")
    logger.info("=" * 50)

    tokenizer, model = load_model(
        args.model_path,
        args.reg_ckpt,
        device
    )

    logger.info("Model loaded successfully.")
    logger.info("=" * 50)
    logger.info("Predicting...")
    df = scan(sequence, positions, model, tokenizer, device)
    df.to_csv(args.output_file, index=False)
    logger.info("=" * 50)
    logger.info(f"Prediction results saved to: {args.output_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main(args)