from pydantic import BaseModel, field_validator, model_validator
from fractions import Fraction
from typing import Any, Dict, Literal, Union, Optional, List
import torch.nn as nn
from kdfp.core.projector.offline.offlineProjector import OfflineProjector
from kdfp.core.projector.online.onlineProjector import OnlineProjector
from kdfp.schemas.configBase import ConfigBase
import os

class LayerReductionConfig(BaseModel):
    name: str
    factor: Union[str, float]

    @field_validator('factor')
    def convert_to_float(cls, v):
        f = float(Fraction(v))
        return f

class DimReductionConfig(BaseModel):
    name: str

class StudentConfigs(BaseModel):
    shallow: LayerReductionConfig
    narrow: DimReductionConfig

class MiniMaxiHyperparam(BaseModel):
    mini: Union[float, int]
    maxi: Union[float, int]

class HyperparamsConfig(BaseModel):
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

class TaskConfig(BaseModel):
    dataset: str
    subset: Optional[str] = None

class TrainingParamsConfig(BaseModel):
    seed: int
    seq_len: int
    num_epochs: int
    reproducible_backend: bool
    num_workers: int
    device: str

# class ProjectorTrainingConfig(BaseModel):
#     dataset: str
#     total_tokens: int

class ReductionHooks(BaseModel):
    shallow: List[str]
    narrow: List[str]

class DistillationHooks(BaseModel):
    teacher: ReductionHooks
    student: ReductionHooks

class ReductionDims(BaseModel):
    shallow: int
    narrow: int

class DistillationDims(BaseModel):
    teacher: ReductionDims
    student: ReductionDims
    
class DatasetConfig(BaseModel):
    name: str
    subset: Optional[str] = None
    split: str = 'train'
    shuffle: bool = False
    token_rename: bool = False

class BaseDirConfig(BaseModel):
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

    def to_config_dict(self) -> Dict[str, Any]:
        config_dict = {
            "cache": self.cache,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "student_scores": self.student_scores,
            "off_proj_model": self.off_proj_model,
        }
        return config_dict

class OffProjTrainArgsConfig(BaseModel):
    dataset: DatasetConfig
    total_tokens: int
    cache_batch_size: int
    n_trials: int
    force_cache_all: bool
    num_gpus_per_trial: float
    num_cpus_per_trial: int

class OffProjParamsConfig(BaseModel):
    alignment_betas: List[float]
    training_args: OffProjTrainArgsConfig

class ExpProjectionConfig(ConfigBase):
    model: Optional[nn.Module] = None
    direction: Optional[Literal["up", "down"]]
    training: Optional[Literal["online", "offline"]]
    offline_projector_alignment_beta: Optional[float]
    offline_projector_training_args: Optional[OffProjTrainArgsConfig]

    def to_config_dict(self) -> Dict[str, Any]:
        config_dict = {
            "direction": self.direction,
            "training": self.training,
            "offline_projector_alignment_beta": self.offline_projector_alignment_beta
        }
        return config_dict