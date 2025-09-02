from typing import Optional, Dict, Any, List
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
import torch
import kdfp.utils.weight_conversion as wc
from kdfp.schemas.configBase import ConfigBase


class LM(ConfigBase):
    model_name: str
    d_model: int
    # model_dir: Optional[str] = None
    tl: Optional[HookedTransformer] = None
    hf: Optional[AutoModelForCausalLM] = None
    layer_reduction_factor: Optional[float] = None
    hook_list: Optional[List[str]] = None
    
    def to_config_dict(self) -> Dict[str, Any]:
        """
        Return a standardized dictionary representation for configuration purposes.
        Excludes the heavy model objects (tl, hf).
        """
        config_dict = {
            'model_name': self.model_name,
            'layer_reduction_factor': self.layer_reduction_factor,
            'hook_list': self.hook_list,
            'd_model': self.d_model,
        }
        return config_dict

    def save_tl(self, save_dir: str) -> None:
        if self.tl is None:
            raise ValueError("No HookedTransformer has been loaded yet. Please make sure you load the model under the 'tl' field for an 'LM' object")
        tl_dir = save_dir + "/tl"
        state_dict = self.tl.state_dict()
        torch.save(state_dict, f"{tl_dir}/state_dict.pt")
        return

    def load_and_save_hf(self, save_dir: str) -> None:
        tl_dir = save_dir + "/tl"
        hf_dir = save_dir + "/hf"
        if self.tl is not None:
            state_dict = self.tl.state_dict()
        else:
            # self.save_tl(save_dir=save_dir)
            state_dict = torch.load(f"{tl_dir}/state_dict.pt")

        if "pythia" in self.model_name:
            converted_weights, _ = wc.convert_hooked_to_neox_weights(state_dict, self.tl.cfg)
        elif "gpt2" in self.model_name:
            converted_weights, _ = wc.convert_hooked_to_gpt2_weights(state_dict, self.tl.cfg)
        else:
            raise NotImplementedError("Only weight conversions for Pythia and GPT-2 models is currently implemented.")
        
        hf = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            state_dict=converted_weights,
        )
        hf.save_pretrained(hf_dir)
        self.hf = hf
        return
