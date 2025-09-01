from pydantic import BaseModel, field_validator
from fractions import Fraction
from typing import Union, Optional, List

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
    layer_reduction: LayerReductionConfig
    dim_reduction: DimReductionConfig

class HyperparamsConfig(BaseModel):
    lr: List[float]
    weight_decay: List[float]
    temperature: List[float]
    max_grad_norm: List[float]
    batch_size: List[int]
    # num_epochs: List[int]

class TaskConfig(BaseModel):
    dataset: str
    subset: Optional[str] = None

class TrainingParamsConfig(BaseModel):
    seed: List[int]
    seq_len: int
    num_epochs: int
    reproducible_backend: bool

class ProjectorTrainingConfig(BaseModel):
    dataset: str
    total_tokens: int

class HookLists(BaseModel):
    shallow: List[str]
    narrow: List[str]

class DistillationHooks(BaseModel):
    teacher: HookLists
    student: HookLists
