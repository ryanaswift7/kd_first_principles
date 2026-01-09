from typing import Optional, Union

from kdfp.schemas.configBase import ConfigBase


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


class FixedParams(ConfigBase):
    teacher: Optional[FixedHparamSet] = None
    pretrained: Optional[DistillPairHparams] = None
    shallow: Optional[DistillPairHparams] = None


class HparamsConfig(ConfigBase):
    fixed: Optional[FixedParams] = None
    tune: Optional[TuneParams] = None
