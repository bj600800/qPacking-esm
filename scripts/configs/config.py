"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/2/14

# Description: Manage configuration datatype and value
# ------------------------------------------------------------------------------
"""
import yaml
from dataclasses import dataclass

@dataclass
class PathConfig:
    model_dir: str
    checkpoints_dir: str
    logging_dir: str
    tokenized_cache_path: str
    fasta_file: str
    pkl_file: str

@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float

@dataclass
class TrainingArgsConfig:
    seed: int
    lr: float
    test_ratio: float
    batch_size: int
    num_epochs: int
    eval_strategy: str
    save_strategy: str
    eval_steps: int
    save_steps: int
    logging_steps: int
    save_total_limit: int
    reporter: str
    metric_for_best_model: str
    greater_is_better: bool

@dataclass
class Config:
    # class type hint
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsConfig

    @staticmethod
    def from_yaml(path: str) -> 'Config':
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        return Config(
            path=PathConfig(**raw['path']),  # path: class instance
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsConfig(**raw['training_args'])
        )

if __name__ == '__main__':
    path = r"/Users/douzhixin/Developer/qPacking/code/scripts/configs/hydrophobic_binary.yaml"
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    for section, section_value in raw.items():
        print(f"[{section}]")
        for k, v in section_value.items():
            print(f"  {k}: {v} ({type(v)})")