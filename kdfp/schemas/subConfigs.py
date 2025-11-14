from pydantic import BaseModel, field_validator, model_validator, field_serializer
from fractions import Fraction
from typing import Any, Dict, Literal, Union, Optional, List
from kdfp.schemas.configBase import ConfigBase
import os
from kdfp.schemas.mixins import DtypeSerializerMixin
from kdfp.utils.enums import (ProjectionDirection, ProjectorTraining, StudentReduction)
import torch

class LayerReductionConfig(ConfigBase):
    name: str
    layer_reduction_factor: Union[str, float]

    @field_validator('layer_reduction_factor')
    def convert_to_float(cls, v):
        f = float(Fraction(v))
        return f

class DimReductionConfig(ConfigBase):
    name: str
    kd_layer_filter_factors: List[Union[str, float]]

    @field_validator('kd_layer_filter_factors')
    def convert_to_float(cls, v):
        try:
            return [float(Fraction(str(f))) for f in v]
        except ValueError as e:
            # This will catch invalid strings like "abc"
            raise ValueError(f"Invalid fraction or float string in list: {e}")
        except TypeError as e:
            # This might catch if the list contains something unexpected like None
            raise TypeError(f"Invalid type in list: {e}")
        
class NumLayersConfig(ConfigBase):
    student_narrow: int
    student_shallow: int
    teacher_narrow: int
    teacher_shallow: int

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

class TaskConfig(ConfigBase):
    dataset_name: str
    subset: Optional[str] = None

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
    ray: str
    lightning: str

class ExpDirConfig(ConfigBase):
    cache: Optional[str] = None
    off_proj_models: Optional[str] = None
    teacher_model: str
    teacher_scores: str
    student_model: str
    student_scores: str
    ray: str
    lightning: str

    @model_validator(mode='after')
    def make_paths_absolute(self):
        self.teacher_model = os.path.abspath(self.teacher_model)
        self.teacher_scores = os.path.abspath(self.teacher_scores)
        self.student_model = os.path.abspath(self.student_model)
        self.student_scores = os.path.abspath(self.student_scores)
        self.ray = os.path.abspath(self.ray)
        self.lightning = os.path.abspath(self.lightning)
        if self.cache:
            self.cache = os.path.abspath(self.cache)
        if self.off_proj_models:
            self.off_proj_models = os.path.abspath(self.off_proj_models)
        return self


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

class ProjectionConfig(ConfigBase):
    # model: Optional[nn.Module] = None
    direction: Optional[ProjectionDirection]
    training: Optional[ProjectorTraining]
    offline_projector_alignment_beta: Optional[float]
    offline_projector_training_args: Optional[OffProjTrainArgsConfig]

    # _heavy_fields_to_exclude = set(["model"])
    
class LMParamsConfig(ConfigBase):
    num_samples: Optional[int] = None
    num_epochs: int
    n_trials: int
    test_batch_size: int

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

class TrainingParamsConfig(ConfigBase, DtypeSerializerMixin):
    get_baselines: bool
    run_or_test: Literal['run', 'test']
    testing_mode: Literal['pl', 'hf']
    seed: int
    seq_len: int
    reproducible_backend: bool
    device: str
    torch_dtype: Union[str, torch.dtype]
    resources: ExperimentResources