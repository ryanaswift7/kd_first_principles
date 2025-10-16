from typing import Optional, Dict, Any, List
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
import torch
import kdfp.utils.weight_conversion as wc
from kdfp.schemas.configBase import ConfigBase
from lightning.pytorch import LightningModule


class LMConfig(ConfigBase):
    model_name: str
    d_model: int
    layer_reduction_factor: Optional[float] = None
    hook_list: Optional[List[str]] = None

