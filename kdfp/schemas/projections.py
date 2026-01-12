from typing import List, Optional

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.dataset import DatasetConfig
from kdfp.utils.enums import ProjectionDirection, ProjectionTraining


class OffProjTrainArgs(ConfigBase):
    dataset: DatasetConfig
    total_tokens: int
    cache_batch_size: int
    num_epochs: int
    n_trials: int
    force_cache_all: bool


class OffProjInstance(ConfigBase):
    alignment_beta: float
    training_args: OffProjTrainArgs


class ProjectionInstance(ConfigBase):
    direction: ProjectionDirection
    training_method: ProjectionTraining
    offline_alignment_beta: Optional[float]


class ProjectionParser(ConfigBase):
    directions: List[ProjectionDirection]
    training_methods: List[ProjectionTraining]
    offline_alignment_betas: List[float]
