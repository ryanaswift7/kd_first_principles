from typing import Optional, Union

from kdfp.schemas.configBase import ConfigBase
from kdfp.utils.enums import StudentType, TrainingType


class MiniMaxiHyperparam(ConfigBase):
    mini: Union[float, int]
    maxi: Union[float, int]


class TuneParams(ConfigBase):
    lr: MiniMaxiHyperparam
    weight_decay: MiniMaxiHyperparam
    temperature: MiniMaxiHyperparam
    max_grad_norm: MiniMaxiHyperparam


class FixedHparamSet(ConfigBase):
    lr: float
    weight_decay: float
    max_grad_norm: float
    temperature: Optional[float] = None


class DistillPairHparams(ConfigBase):
    ft: Optional[FixedHparamSet] = None
    kd: Optional[FixedHparamSet] = None


class FixedOffProjHparamSet(ConfigBase):
    lr: float
    weight_decay: float
    tied_weights: bool


class FixedParams(ConfigBase):
    teacher: Optional[FixedHparamSet] = None
    pretrained: Optional[DistillPairHparams] = None
    shallow: Optional[DistillPairHparams] = None
    off_proj: Optional[FixedOffProjHparamSet] = None


class HparamsConfig(ConfigBase):
    fixed: Optional[FixedParams] = None
    tune: Optional[TuneParams] = None


def get_fixed_student_params(
    hyperparams: HparamsConfig, student_type: StudentType, training_type: TrainingType
):
    if not hyperparams.fixed:
        return None

    # Map types to the specific config objects
    try:
        student_group = {
            StudentType.SHALLOW: hyperparams.fixed.shallow,
            StudentType.PRETRAINED: hyperparams.fixed.pretrained,
        }.get(student_type)

        if not student_group:
            return None

        return {
            TrainingType.FT: student_group.ft,
            TrainingType.KD: student_group.kd,
        }.get(training_type)
    except AttributeError:
        return None
