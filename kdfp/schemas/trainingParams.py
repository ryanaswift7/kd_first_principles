from typing import List, Literal, Optional, Union

import torch
from pydantic import computed_field

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.mixins import DtypeSerializerMixin


class ResourceConfig(ConfigBase):
    num_data_workers: int
    num_ray_workers: int
    num_cpus_per_trial: int
    num_gpus_per_trial: float


class ExperimentResources(ConfigBase):
    max_failures: int
    num_available_gpus: int
    projector: ResourceConfig
    teacher: ResourceConfig
    student: ResourceConfig


class TrainingControls(ConfigBase):
    finetune_teacher: bool
    get_baselines: bool
    run_or_test: Literal["run", "test"]
    testing_mode: Literal["pl", "hf"]


class ObjectiveParams(ConfigBase):
    weight_variants: List[str]
    jsd_betas: List[float]


class WandbParams(ConfigBase):
    project: str
    group: str


class TrainingParamsConfig(ConfigBase, DtypeSerializerMixin):
    seeds: List[int]
    seq_len: int
    train_batch_size: int
    test_batch_size: int
    num_samples: Optional[int] = None
    num_epochs: int
    n_trials: int
    reproducible_backend: bool
    device: str
    torch_dtype: Union[str, torch.dtype]
    controls: TrainingControls
    objective: ObjectiveParams
    resources: ExperimentResources
    wandb: WandbParams

    @computed_field
    @property
    def default_seed(self) -> int:
        return self.seeds[0]
