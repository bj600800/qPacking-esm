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
from qpacking.utils import logger

logger = logger.setup_log(name=__name__)


@dataclass
class PathConfig:
    model_dir: str
    checkpoints_dir: str
    logging_dir: str
    tokenized_cache_path: str
    fasta_file: str
    pkl_file: str
    model_src: str

@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float

@dataclass
class TrainingArgsHydrophobicConfig:
    seed: int
    lr: float
    num_epochs: int
    test_ratio: float
    batch_size: int
    eval_strategy: str
    save_strategy: str
    save_steps: str
    save_total_limit: int
    logging_strategy: str
    eval_steps: int
    logging_steps: int
    reporter: str
    metric_for_best_model: str
    greater_is_better: bool

@dataclass
class TrainingArgsHydrophobicBinaryConfig(TrainingArgsHydrophobicConfig):
    num_class: int

@dataclass
class TrainingArgsHydrophobicContrastiveConfig(TrainingArgsHydrophobicConfig):
    proj_dim: int

@dataclass
class PathConfigFitness(PathConfig):
    model_src: str

@dataclass
class TrainingArgsFitnessConfig(TrainingArgsHydrophobicConfig):
    unfreeze_last_n: int
    emb_src: str



@dataclass
class ConfigHydrophobicBinary:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsHydrophobicBinaryConfig

@dataclass
class ConfigHydrophobicContrastive:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsHydrophobicContrastiveConfig

@dataclass
class ConfigDegree:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsHydrophobicConfig

@dataclass
class ConfigArea:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsHydrophobicConfig

@dataclass
class ConfigRsa:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsHydrophobicConfig

@dataclass
class ConfigOrder:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsHydrophobicConfig

@dataclass
class ConfigCentrality:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsHydrophobicConfig

@dataclass
class ConfigFitness:
    path: PathConfigFitness
    training_args: TrainingArgsFitnessConfig

def from_yaml(path: str, task: str):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    if task == "hydrophobic_binary":
        return ConfigHydrophobicBinary(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsHydrophobicBinaryConfig(**raw['training_args'])
        )
    elif task == "hydrophobic_contrastive":
        return ConfigHydrophobicContrastive(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsHydrophobicContrastiveConfig(**raw['training_args'])
        )
    elif task == "degree":
        return ConfigDegree(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsHydrophobicConfig(**raw['training_args'])
        )

    elif task == "area":
        return ConfigArea(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsHydrophobicConfig(**raw['training_args'])
        )

    elif task == "rsa":
        return ConfigRsa(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsHydrophobicConfig(**raw['training_args'])
        )

    elif task == "order":
        return ConfigOrder(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsHydrophobicConfig(**raw['training_args'])
        )

    elif task == "centrality":
        return ConfigCentrality(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsHydrophobicConfig(**raw['training_args'])
        )

    elif task == "fitness":
        return ConfigFitness(
            path=PathConfigFitness(**raw['path']),
            training_args=TrainingArgsFitnessConfig(**raw['training_args'])
        )

    else:
        raise ValueError(f"Unsupported task type: {task}")


class ConfigLogger:
    """
    A class to log configuration details for different model tasks.
    """
    def __init__(self, config, task: str, logger=logger):
        self.config = config
        self.task = task.lower()
        self.logger = logger

    def log(self):
        self.logger.info(f"\n{'='*10} [Task: {self.task}] Config Summary {'='*10}")
        self._log_common()

        if self.task == "hydrophobic_binary":
            self._log_hydrophobic_binary()
        elif self.task == "hydrophobic_contrastive":
            self._log_hydrophobic_contrastive()
        elif self.task == "degree":
            self._log_degree()
        elif self.task == 'area':
            self._log_area()
        elif self.task == 'rsa':
            self._log_rsa()
        elif self.task == 'order':
            self._log_order()
        elif self.task == 'centrality':
            self._log_centrality()
        elif self.task == "fitness":
            self._log_fitness()
        else:
            self.logger.warning(f"Unknown task: {self.task}. Logging only common parameters.")
        self.logger.info(f"\n{'=' * 10} End of Config Summary {'=' * 10}")

    def _log_common(self):
        cfg = self.config
        self.logger.info(f"{'='*10}[Path]{'='*10}")
        for k, v in vars(cfg.path).items():
            self.logger.info(f"{k}: {v}")

        self.logger.info(f"{'='*10}[Training Args]{'='*10}")
        for k, v in vars(cfg.training_args).items():
            self.logger.info(f"{k}: {v}")

        if self.task != "fitness":
            self.logger.info(f"{'='*10}[LoRA]{'='*10}")
            for k, v in vars(cfg.lora).items():
                self.logger.info(f"{k}: {v}")

    def _log_hydrophobic_binary(self):
        self.logger.info("[Hydrophobic-binary Task Specific Config]")
        self.logger.info(f"num_class: {self.config.training_args.num_class}")


    def _log_hydrophobic_contrastive(self):
        self.logger.info("[Hydrophobic-contrastive Task Specific Config]")
        self.logger.info(f"proj_dim: {self.config.training_args.proj_dim}")


    def _log_degree(self):
        pass

    def _log_area(self):
        pass

    def _log_rsa(self):
        pass

    def _log_order(self):
        pass

    def _log_centrality(self):
        pass

    def _log_fitness(self):
        self.logger.info("[Fitness Task Specific Config]")
        self.logger.info(f"model_src: {self.config.path.model_src}")
        self.logger.info(f"unfreeze_last_n: {self.config.training_args.unfreeze_last_n}")
        self.logger.info(f"emb_src: {self.config.training_args.emb_src}")