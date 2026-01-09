from typing import List, Optional

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.dataset import DatasetConfig
from kdfp.utils.enums import ProjectionDirection, ProjectorTraining


class OffProjTrainArgsConfig(ConfigBase):
    dataset: DatasetConfig
    total_tokens: int
    cache_batch_size: int
    num_epochs: int
    n_trials: int
    force_cache_all: bool


class OffProjParamsConfig(ConfigBase):
    alignment_betas: List[float]
    training_args: OffProjTrainArgsConfig


class OffProjInstance(ConfigBase):
    alignment_beta: float
    training_args: OffProjTrainArgsConfig


class ProjectionInstance(ConfigBase):
    direction: ProjectionDirection
    training: ProjectorTraining
    offline: Optional[OffProjInstance] = None


class ProjectionConfig(ConfigBase):
    directions: List[ProjectionDirection]
    training: List[ProjectorTraining]
    offline_params: OffProjParamsConfig
