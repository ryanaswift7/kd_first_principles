from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
from kdfp.schemas.experimentConfig import ExperimentConfig, TaskSpec, LM
from transformer_lens import HookedTransformer
from kdfp.utils.shallow_model_loading import get_indices_and_num_layers_to_keep
from kdfp.schemas.objective_weights import OBJECTIVE_WEIGHT_VARIANTS_MAP
from kdfp.schemas.subConfigs import (StudentConfigs,
                                       ProjectorTrainingConfig,
                                       TrainingParamsConfig,
                                       HyperparamsConfig,
                                       TaskConfig,
                                       DistillationHooks,
                                       HookLists)


class GlobalConfig(BaseModel):
    teacher_name: str
    students: StudentConfigs
    projector_training: ProjectorTrainingConfig
    training_params: TrainingParamsConfig
    hook_types: Dict[str, List[str]] = Field(..., description="Keys 'single' and 'per_layer' with lists of hook names")
    projection: Dict[str, List[str]] = Field(..., description="Keys 'directions' and 'training_methods'")
    offline_projector_alignment_betas: List[float]
    objective_weight_variants: List[str]
    jsd_betas: List[float]
    hyperparams: HyperparamsConfig
    tasks: Dict[str, TaskConfig] # A dictionary where the key is the task name (e.g., 'hellaswag') and the value is its config
    hooks: Optional[DistillationHooks] = None

    # Optional: Validator to ensure hook point keys are expected
    @field_validator('hook_types')
    def validate_hook_keys(cls, v):
        allowed_keys = {'single', 'per_layer'}
        if not all(key in allowed_keys for key in v.keys()):
            raise ValueError(f'hook_types can only contain the keys: {allowed_keys}')
        return v
    
    @model_validator(mode='after')
    def generate_both_hook_lists(self) -> 'GlobalConfig':
        # self.hooks = DistillationHooks()
        tn = self._generate_hook_list(model_name=self.teacher_name)
        ts = self._generate_hook_list(model_name=self.teacher_name,
                                                      layer_reduction_factor=self.students.layer_reduction.factor,
                                                      selection_method='linspace')
        """
        - The shallow student is first n layers because once we actually load in the shallow student,
          the first n layers will be the entire student model.
        - This way, the linspaced teacher layers match up with all of the student layers
        """
        sn = self._generate_hook_list(model_name=self.students.dim_reduction.name)
        ss = self._generate_hook_list(model_name=self.students.layer_reduction.name,
                                                      layer_reduction_factor=self.students.layer_reduction.factor,
                                                      selection_method='first_n')
        
        teacher_hooks = HookLists(narrow=tn, shallow=ts)
        student_hooks = HookLists(narrow=sn, shallow=ss)

        self.hooks = DistillationHooks(teacher=teacher_hooks, student=student_hooks)
        return self

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
    
    def _generate_hook_list(self, model_name: str,
                            layer_reduction_factor: Optional[float] = None,
                            selection_method: Optional[Literal['linspace', 'first_n']] = None) -> List[str]:
        lm = HookedTransformer.from_pretrained(model_name)
        if layer_reduction_factor is not None:
            assert selection_method, "You must supply a selection method if performing layer reduction"
            layer_idx_list, _ = get_indices_and_num_layers_to_keep(total_layers=lm.cfg.n_layers,
                                                         layer_reduction_factor=layer_reduction_factor,
                                                         selection_method=selection_method)
        else:
            layer_idx_list = list(range(lm.cfg.n_layers))

        hooks = []
        hooks.extend(self.hook_types["single"])
        for idx in layer_idx_list:
            for hp in self.hook_types["per_layer"]:
                hooks.append(f"blocks.{idx}.{hp}")
        return hooks

    def generate_experiment_list(self, debug: bool = False) -> List[ExperimentConfig]:
        tasks = self.generate_task_list()

        experiment_list = []

        for proj_dir in self.projection["directions"]:
            for proj_train in self.projection["training_methods"]: 
                for obj_weight_variant in self.objective_weight_variants:
                    for jsd_beta in self.jsd_betas:
                            for off_proj_align_beta in self.offline_projector_alignment_betas:
                                for reduction in ["shallow", "narrow"]:
                                    if reduction == "narrow":
                                        proj_str = f"{proj_dir}_{proj_train}"
                                        if proj_train == "offline":
                                            proj_str += f"_align_{off_proj_align_beta}"
                                        proj_str += '-'
                                        teacher = LM(model_name=self.teacher_name,
                                                     hook_list=self.hooks.teacher.narrow)
                                        student = LM(model_name=self.students.dim_reduction.name,
                                                     hook_list=self.hooks.student.narrow)
                                        # hook_list = self.teacher_narrow_hooks
                                        projection_direction = proj_dir
                                        projection_training = proj_train
                                        offline_projector_alignment_beta = off_proj_align_beta

                                    elif reduction == "shallow":
                                        proj_str = ""
                                        teacher = LM(model_name=self.teacher_name,
                                                     hook_list=self.hooks.teacher.shallow)
                                        student = LM(model_name=self.students.layer_reduction.name,
                                                     hook_list=self.hooks.student.shallow)
                                        projection_direction = None
                                        projection_training = None
                                        offline_projector_alignment_beta = None

                                    else:
                                        raise ValueError("Invalid reduction specified. Must be either 'layer' or 'dim'.")


                                    exp_name = f"{reduction}-{proj_str}{obj_weight_variant}-jsd_{jsd_beta}"
                                    exp = ExperimentConfig(
                                        name=exp_name,
                                        reduction=reduction,
                                        teacher=teacher,
                                        student=student,
                                        training_params=self.training_params,
                                        projection_direction=projection_direction,
                                        projection_training=projection_training,
                                        offline_projector_alignment_beta=offline_projector_alignment_beta,
                                        # hook_list=hook_list,
                                        objective_weight_variant=obj_weight_variant,
                                        objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP[obj_weight_variant],
                                        jsd_beta=jsd_beta,
                                        hyperparams=self.hyperparams,
                                        tasks=tasks,
                                    )
                                    experiment_list.append(exp)
                                    # exp_names.append(exp_name)

        # need to add in baseline experiments
        ft_shallow_student = ExperimentConfig(
            name="shallow_ft",
            reduction="shallow",
            teacher=LM(model_name=self.teacher_name),
            student=LM(model_name=self.students.layer_reduction.name,
                       layer_reduction_factor=self.students.layer_reduction.factor),
            training_params=self.training_params,
            projection_direction=None,
            projection_training=None,
            offline_projector_alignment_beta=None,
            # hook_list=hook_list,
            objective_weight_variant="finetuning",
            objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP["finetuning"],
            jsd_beta=None,
            hyperparams=self.hyperparams,
            tasks=tasks,
        )
        ft_narrow_student = ExperimentConfig(
            name="narrow_ft",
            reduction="narrow",
            teacher=LM(model_name=self.teacher_name),
            student=LM(model_name=self.students.dim_reduction.name),
            training_params=self.training_params,
            projection_direction=None,
            projection_training=None,
            offline_projector_alignment_beta=None,
            # hook_list=hook_list,
            objective_weight_variant="finetuning",
            objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP["finetuning"],
            jsd_beta=None,
            hyperparams=self.hyperparams,
            tasks=tasks,
        )

        experiment_list.extend([ft_shallow_student, ft_narrow_student])
        # exp_names.extend([ft_shallow_student.name, ft_narrow_student.name])

        if debug:
            print(f"Total experiments created: {len(experiment_list)}")

        # NOTE: since I didn't want to build the logic into the loop, duplicate experiments will be created
        # - ConfigBase provides the methods to allow me to hash and compare experiments
        # - I use this to filter out the duplicates
        unique_experiment_list = list(set(experiment_list))
        exp_names = [e.name for e in unique_experiment_list]

        return experiment_list, sorted(exp_names)

