"""
TSNE Visualization Suite for ESM2 Fine-tuning
Author: Dou Zhixin
Purpose:
1. Visualize embedding distribution before and after fine-tuning
2. Show predicted positions from classification head cluster together
3. Show embedding drift magnitude (for regression values)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from transformers import EsmModel, EsmTokenizer
from peft import PeftModel, PeftConfig
import torch.nn as nn
import os
from qpacking_esm.model.heads import ClassificationHead, RegressionHead

device = "cuda" if torch.cuda.is_available() else "cpu"

class TaskModel(nn.Module):
    def __init__(self, backbone, head_path, task_type="classification"):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        hidden_size = backbone.config.hidden_size

        if task_type == "classification":
            self.head = ClassificationHead(hidden_size, 2)
            state_dict = torch.load(head_path, map_location="cpu")
            state_dict_linear = {k.replace("classifier.", ""): v for k, v in state_dict.items() if k.startswith("classifier.")}
            self.head.classifier.load_state_dict(state_dict_linear)
        elif task_type == "regression":
            self.head = RegressionHead(hidden_size)
            state_dict = torch.load(head_path, map_location="cpu")
            if all(k.startswith("regressor.") for k in state_dict.keys()):
                state_dict_linear = {k.replace("regressor.", ""): v for k, v in state_dict.items()}
                self.head.regressor.load_state_dict(state_dict_linear)
            else:
                self.head.regressor.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unknown task_type {task_type}")

    def forward(self, input_ids, attention_mask=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden[:, 1:-1, :]  # remove CLS/EOS
        return self.head(hidden)

    def get_embedding(self, input_ids, attention_mask=None):
        with torch.no_grad():
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            hidden = hidden[:, 1:-1, :]
        return hidden

def load_model(path, is_finetuned=False):
    if not is_finetuned:
        model = EsmModel.from_pretrained(path, add_pooling_layer=False)
        tokenizer = EsmTokenizer.from_pretrained(path)
    else:
        peft_config = PeftConfig.from_pretrained(path)
        base_model = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
        model = PeftModel.from_pretrained(base_model, path)
        tokenizer = EsmTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    return model.to(device).eval(), tokenizer

def extract_pred_values_classification(clf_model, tokenizer, seqs, max_per_seq=400):
    preds = []
    embs  = []
    for seq in seqs:
        inp = tokenizer(seq, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}

        with torch.no_grad():
            hidden = clf_model.get_embedding(**inp)[0]  # backbone embedding
            if clf_model.head is not None:
                logits = clf_model.head(hidden)
                pred = torch.argmax(logits.logits, dim=-1)
            else:
                pred = torch.zeros(hidden.shape[0], dtype=torch.long)

        if len(hidden) > max_per_seq:
            idx = torch.randperm(len(hidden))[:max_per_seq]
            hidden = hidden[idx]
            pred = pred[idx]

        preds.append(pred.cpu().numpy())
        embs.append(hidden.cpu())
    return np.concatenate(preds, axis=0), torch.cat(embs, dim=0).numpy()

def extract_pred_values_regression(reg_model, tokenizer, seqs, max_per_seq=400):
    vals = []
    embs = []
    for seq in seqs:
        inp = tokenizer(seq, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}

        with torch.no_grad():
            hidden = reg_model.get_embedding(**inp)[0]
            if reg_model.head is not None:
                output = reg_model.head(hidden)
                pred = output.logits.squeeze(-1)
            else:
                pred = torch.zeros(hidden.shape[0], dtype=torch.float)

        if len(hidden) > max_per_seq:
            idx = torch.randperm(len(hidden))[:max_per_seq]
            hidden = hidden[idx]
            pred = pred[idx]

        vals.append(pred.cpu().numpy())
        embs.append(hidden.cpu())
    return np.concatenate(vals, axis=0), torch.cat(embs, dim=0).numpy()

def run_tsne(X, perplex=60):
    tsne = TSNE(n_components=2, perplexity=perplex, learning_rate="auto", init="pca", random_state=200)
    return tsne.fit_transform(X)

def plot_tsne(X2d, plot_path, labels=None, values=None, title="", legend_type="pred", is_regression=False):
    plt.figure(figsize=(7, 7))
    if is_regression and values is not None:
        sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=values, cmap="plasma", s=8, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, label="Regression value")
    elif labels is not None:
        colors = ['#20B2AA', '#C71585']
        plt.scatter(
            X2d[:, 0], X2d[:, 1],
            c=[colors[l] for l in labels],
            s=8, alpha=0.8, edgecolors='none'
        )

        if legend_type == "pred":
            class0 = mpatches.Patch(color=colors[0], label='Surface')
            class1 = mpatches.Patch(color=colors[1], label='Interior')
            plt.legend(handles=[class1, class0])
    else:
        plt.scatter(X2d[:, 0], X2d[:, 1], s=8, alpha=0.6, edgecolors='none')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.show()

if __name__ == "__main__":
    official_dir  = "/Users/douzhixin/Developer/qPacking-esm/data/checkpoints/esm2_t30_150M_UR50D"
    finetuned_dir = "/Users/douzhixin/Developer/qPacking-esm/data/checkpoints/hpd/bsa/bsa-lora1/checkpoint-3500"
    task = "bsa"
    fasta = "/Users/douzhixin/Developer/qPacking-esm/data/test/sequence/seq.fasta"
    plot_dir = r"/Users/douzhixin/Developer/qPacking-esm/figure/python/tsne"
    seqs = [l.strip() for l in open(fasta) if not l.startswith(">")]

    # Load pre-trained and fine-tuned backbones
    model_pre, tok_pre   = load_model(official_dir, is_finetuned=False)
    model_post, tok_post = load_model(finetuned_dir, is_finetuned=True)

    head_path = os.path.join(finetuned_dir, "task_head.pt")


    ## classification task
    # clf_model_pre = TaskModel(model_pre, head_path=head_path, task_type="classification")
    # pred_before, emb_before_clf = extract_pred_values_classification(clf_model_pre, tok_pre, seqs)
    # X2d_pred_before = run_tsne(emb_before_clf)
    # plot_path_pretrained = os.path.join(plot_dir, f"tsne_{task}_pretrained.png")
    #
    # plot_tsne(X2d_pred_before, labels=pred_before, title="Pre-trained Backbone + Trained Head", legend_type="pred", plot_path=plot_path_pretrained)
    #
    # # Fine-tuned backbone + trained head
    # clf_model_post = TaskModel(model_post, head_path=head_path, task_type="classification")
    # pred_after, emb_after_clf = extract_pred_values_classification(clf_model_post, tok_post, seqs)
    # X2d_pred_after = run_tsne(emb_after_clf)
    # plot_path_finetuned = os.path.join(plot_dir, f"tsne_{task}_finetuned.png")
    # plot_tsne(X2d_pred_after, labels=pred_after, title="Fine-tuned Backbone + Trained Head", legend_type="pred", plot_path=plot_path_finetuned)

    ## regression task
    reg_model_pre = TaskModel(model_pre, head_path=head_path, task_type="regression")
    val_before, emb_before_reg = extract_pred_values_regression(reg_model_pre, tok_pre, seqs)
    X2d_reg_before = run_tsne(emb_before_reg)
    plot_path_pretrained = os.path.join(plot_dir, f"tsne_{task}_pretrained.png")
    plot_tsne(X2d_reg_before, values=val_before, title="Pre-trained Backbone + Trained Head (Regression)", is_regression=True, plot_path=plot_path_pretrained)

    reg_model_post = TaskModel(model_post, head_path=head_path, task_type="regression")
    val_after, emb_after_reg = extract_pred_values_regression(reg_model_post, tok_post, seqs)
    X2d_reg_after = run_tsne(emb_after_reg)
    plot_path_finetuned = os.path.join(plot_dir, f"tsne_{task}_finetuned.png")
    plot_tsne(X2d_reg_after, values=val_after, title="Fine-tuned Backbone + Trained Head (Regression)", is_regression=True, plot_path=plot_path_finetuned)
