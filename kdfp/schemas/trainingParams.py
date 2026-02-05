from typing import Literal, Optional

import torch
from pydantic import computed_field

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
    total_tokens: Optional[int] = None
    num_epochs: int
    n_trials: int
    reproducible_backend: bool
    device: str
    torch_dtype: torch.dtype
    controls: TrainingControls
    resources: ExperimentResources
    offline_projection: OffProjTrainArgs
    wandb: WandbParams

    @computed_field
    @property
    def num_samples(self) -> Optional[int]:
        if self.total_tokens and self.seq_len:
            return self.total_tokens // self.seq_len
        else:
            return None
