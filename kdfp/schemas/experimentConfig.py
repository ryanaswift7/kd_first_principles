from pydantic import BaseModel
from typing import Optional, Literal, List, Union
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
import torch
import utils.weight_conversion as wc
from utils.shallow_model_loading import load_shallow
from utils.objective_weights import ObjectiveWeights


class TaskSpec(BaseModel): 
    task_name: str
    dataset: str
    subset: Optional[str] = None
    num_fewshot: int = 0
    batch_size: Union[int, str] = "auto"

class LM(BaseModel):
    model_name: str
    tl: Optional[HookedTransformer] = None
    hf: Optional[AutoModelForCausalLM] = None
    layer_reduction_factor: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True

    def save_tl(self, save_dir: str) -> None:
        if self.tl is None:
            raise ValueError("No HookedTransformer has been loaded yet. Please make sure you load the model under the 'tl' field for an 'LM' object")
        tl_dir = save_dir + "tl/"
        state_dict = self.tl.state_dict()
        torch.save(state_dict, f"{tl_dir}/state_dict.pt")
        return

    def load_and_save_hf(self, save_dir: str) -> None:
        tl_dir = save_dir + "tl/"
        hf_dir = save_dir + "hf/"
        if self.tl is not None:
            state_dict = self.tl.state_dict()
        else:
            state_dict = torch.load(f"{tl_dir}state_dict.pt")

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


class ExperimentConfig(BaseModel):
    experiment_name: str
    reduction: Literal["layer", "dim"]
    teacher: LM
    student: LM
    projection_direction: Optional[Literal["up", "down"]] = None
    projection_training: Optional[Literal["online", "offline"]] = None
    hook_list: List[str]
    objective_weight_variant: str
    objective_weights: ObjectiveWeights
    tasks: List[TaskSpec]

    class Config:
        arbitrary_types_allowed = True


