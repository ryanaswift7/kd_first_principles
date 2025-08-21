from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
import yaml
from experimentConfig import ExperimentConfig, TaskSpec, LM
from transformer_lens import HookedTransformer
from utils.shallow_model_loading import get_indices_and_layers_to_keep
from utils.objective_weights import OBJECTIVE_WEIGHT_VARIANTS_MAP

class LayerReductionConfig(BaseModel):
    name: str
    factor: float

class DimReductionConfig(BaseModel):
    name: str

class StudentConfigs(BaseModel):
    layer_reduction: LayerReductionConfig
    dim_reduction: DimReductionConfig

class HyperparamRange(BaseModel):
    min: float
    max: float
    log: bool = False

class HyperparamsConfig(BaseModel):
    lr: HyperparamRange
    weight_decay: HyperparamRange
    temperature: HyperparamRange
    max_grad_norm: HyperparamRange

class TaskConfig(BaseModel):
    dataset: str
    subset: Optional[str] = None


class GlobalConfig(BaseModel):
    teacher_name: str
    students: StudentConfigs
    hook_points: Dict[str, List[str]] = Field(..., description="Keys 'single' and 'per_layer' with lists of hook names")
    projection: Dict[str, List[str]] = Field(..., description="Keys 'directions' and 'training_methods'")
    objective_weight_variants: List[str]
    hyperparams: HyperparamsConfig
    jsd_betas: List[float]
    tasks: Dict[str, TaskConfig] # A dictionary where the key is the task name (e.g., 'hellaswag') and the value is its config

    # Optional: Validator to ensure hook point keys are expected
    @field_validator('hook_points')
    def validate_hook_keys(cls, v):
        allowed_keys = {'single', 'per_layer'}
        if not all(key in allowed_keys for key in v.keys()):
            raise ValueError(f'hook_points can only contain the keys: {allowed_keys}')
        return v

    # Class method to load from a YAML file
    @classmethod
    def from_yaml(cls, file_path: str) -> 'GlobalConfig':
        with open(file_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        return cls(**raw_config)

    # Class method to load from a YAML string (useful for testing)
    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> 'GlobalConfig':
        raw_config = yaml.safe_load(yaml_str)
        return cls(**raw_config)
    
    def generate_task_list(self) -> List[TaskSpec]:
        tasks = []
        for k, v in self.tasks.items():
            t = TaskSpec(task_name=k, dataset=v.dataset, subset=v.subset)
            tasks.append(t)
        return tasks
    
    def generate_hook_list(self, student_name: str, layer_reduction_factor: Optional[float] = None) -> List[str]:
        lm = HookedTransformer.from_pretrained(student_name)
        if layer_reduction_factor is not None:
            _, n_layers = get_indices_and_layers_to_keep(total_layers=lm.cfg.n_layers, layer_reduction_factor=layer_reduction_factor)
        else:
            n_layers = lm.cfg.n_layers

        hooks = []
        hooks.extend(self.hook_points["single"])
        for i in range(n_layers):
            for hp in self.hook_points["per_layer"]:
                hooks.append(f"blocks.{i}.{hp}")
        return hooks

    def generate_experiment_list(self) -> List[ExperimentConfig]:
        tasks = self.generate_task_list()

        experiment_list = []
        exp_names = []

        for proj_dir in self.projection["directions"]:
            for proj_train in self.projection["training_methods"]: 
                for obj_weight_variant in self.objective_weight_variants:
                    for beta in self.jsd_betas:
                        for reduction in ["layer", "dim"]:
                            if reduction == "dim":
                                proj_str = f"{proj_dir}_{proj_train}-"
                                student = LM(model_name=self.students.dim_reduction.name)
                                hook_list = self.generate_hook_list(student_name=student.model_name)

                            elif reduction == "layer":
                                proj_str = ""
                                student = LM(model_name=self.students.layer_reduction.name,
                                             layer_reduction_factor=self.students.layer_reduction.factor)
                                hook_list = self.generate_hook_list(student_name=student.model_name,
                                                                    layer_reduction_factor=self.students.layer_reduction.factor)
                            else:
                                raise ValueError("Invalid reduction specified. Must be either 'layer' or 'dim'.")


                            exp_name = f"{reduction}-{proj_str}{obj_weight_variant}-jsd_{beta}"
                            exp = ExperimentConfig(
                                experiment_name=exp_name,
                                reduction=reduction,
                                teacher=LM(model_name=self.teacher_name),
                                student=student,
                                projection_direction=proj_dir,
                                projection_training=proj_train,
                                hook_list=hook_list,
                                objective_weight_variant=obj_weight_variant,
                                objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP[obj_weight_variant],
                                tasks=tasks,
                            )
                            experiment_list.append(exp)
                            exp_names.append(exp_name)

        return experiment_list, exp_names

