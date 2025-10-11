from pydantic import BaseModel, field_validator, model_validator, field_serializer
from fractions import Fraction
from typing import Any, Dict, Literal, Union, Optional, List
import torch.nn as nn
from kdfp.core.projector.offline.offlineProjector import OfflineProjector
from kdfp.core.projector.online.onlineProjector import OnlineProjector
from kdfp.schemas.configBase import ConfigBase
import os
from kdfp.utils.enums import (ProjectionDirection, ProjectorTraining, StudentReduction)
import torch

class LayerReductionConfig(ConfigBase):
    name: str
    factor: Union[str, float]

    @field_validator('factor')
    def convert_to_float(cls, v):
        f = float(Fraction(v))
        return f

class DimReductionConfig(ConfigBase):
    name: str

class StudentConfigs(ConfigBase):
    shallow: LayerReductionConfig
    narrow: DimReductionConfig

class MiniMaxiHyperparam(ConfigBase):
    mini: Union[float, int]
    maxi: Union[float, int]

class HyperparamsConfig(ConfigBase):
    lr : MiniMaxiHyperparam
    weight_decay: MiniMaxiHyperparam
    temperature: MiniMaxiHyperparam
    max_grad_norm: MiniMaxiHyperparam
    batch_sizes: List[int]
    # lr: List[float]
    # weight_decay: List[float]
    # temperature: List[float]
    # max_grad_norm: List[float]
    # batch_size: List[int]

class TaskConfig(ConfigBase):
    dataset_name: str
    subset: Optional[str] = None

class TrainingParamsConfig(ConfigBase):
    seed: int
    seq_len: int
    # num_epochs: int
    reproducible_backend: bool
    num_data_workers: int
    device: str

# class ProjectorTrainingConfig(BaseModel):
#     dataset: str
#     total_tokens: int

class ReductionHooks(ConfigBase):
    shallow: List[str]
    narrow: List[str]

class DistillationHooks(ConfigBase):
    teacher: ReductionHooks
    student: ReductionHooks

class ReductionDims(ConfigBase):
    shallow: int
    narrow: int

class DistillationDims(ConfigBase):
    teacher: ReductionDims
    student: ReductionDims
    
class DatasetConfig(ConfigBase):
    name: str
    subset: Optional[str] = None
    split: str = 'train'
    shuffle: bool = False
    token_rename: bool = False

class BaseDirConfig(ConfigBase):
    root: str
    cache: str
    models: str
    scores: str

class ExpDirConfig(ConfigBase):
    cache: Optional[str] = None
    off_proj_model: Optional[str] = None
    teacher_model: str
    student_model: str
    student_scores: str

    @model_validator(mode='after')
    def make_paths_absolute(self):
        self.teacher_model = os.path.abspath(self.teacher_model)
        self.student_model = os.path.abspath(self.student_model)
        self.student_scores = os.path.abspath(self.student_scores)
        if self.cache:
            self.cache = os.path.abspath(self.cache)
        if self.off_proj_model:
            self.off_proj_model = os.path.abspath(self.off_proj_model)
        return self


class OffProjTrainArgsConfig(ConfigBase):
    dataset: DatasetConfig
    total_tokens: int
    cache_batch_size: int
    num_epochs: int
    n_trials: int
    force_cache_all: bool
    num_gpus_per_trial: float
    num_cpus_per_trial: int

class OffProjParamsConfig(ConfigBase):
    alignment_betas: List[float]
    training_args: OffProjTrainArgsConfig

class ExpProjectionConfig(ConfigBase):
    model: Optional[nn.Module] = None
    direction: Optional[ProjectionDirection]
    training: Optional[ProjectorTraining]
    offline_projector_alignment_beta: Optional[float]
    offline_projector_training_args: Optional[OffProjTrainArgsConfig]
    
class LMParamsConfig(ConfigBase):
    num_samples: Optional[int] = None
    num_epochs: int
    n_trials: int
    num_gpus_per_trial: float
    num_cpus_per_trial: float

class DtypeSerializerMixin:
    """A mixin to provide serialization for a 'dtype' field."""
    @field_serializer('torch_dtype')
    def serialize_dtype(self, torch_dtype: torch.dtype) -> str:
        # Converts torch.float32 -> "float32"
        return str(torch_dtype).split('.')[-1]
