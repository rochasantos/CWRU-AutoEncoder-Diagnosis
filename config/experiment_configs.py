from dataclasses import dataclass
from typing import Any

@dataclass
class ExperimentKFoldConfig:
    root_dir: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_repetitions: int
    model_factory: Any
    experiment_key: str
    title: str
    device: str = 'cuda'

@dataclass
class ExperimentPretrainConfig:
    model_factory: Any
    dataset_dir: str
    lr: float
    num_epochs: int
    device='cuda'