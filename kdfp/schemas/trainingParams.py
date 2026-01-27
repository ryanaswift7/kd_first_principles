from typing import Literal, Optional

import torch

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.dataset import DatasetConfig
from kdfp.schemas.mixins import DtypeSerializerMixin
from kdfp.schemas.projections import OffProjTrainArgs


class Resources(ConfigBase):
    num_data_workers: int
    num_ray_workers: int
    num_cpus_per_trial: int
    num_gpus_per_trial: float


class ExperimentResources(ConfigBase):
    max_failures: int
    num_available_gpus: int
    stream_task_data: bool
    projector: Resources
    teacher: Resources
    student: Resources


class TrainingControls(ConfigBase):
    finetune_teacher: bool
    get_baselines: bool
    run_or_test: Literal["run", "test"]


class WandbParams(ConfigBase):
    project: str
    group: str


class TrainingParams(ConfigBase, DtypeSerializerMixin):
    dataset: DatasetConfig
    seq_len: int
    train_batch_size: int
    test_batch_size: int
    num_samples: Optional[int] = None
    num_epochs: int
    n_trials: int
    reproducible_backend: bool
    device: str
    torch_dtype: torch.dtype
    controls: TrainingControls
    resources: ExperimentResources
    offline_projection: OffProjTrainArgs
    wandb: WandbParams
