from typing import List, Optional, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
import torch
import lightning.pytorch as pl
import pandas as pd

from kdfp.core.baseline import BaselineModule, BaselineModuleConfig
from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.experimentConfig import ExperimentConfig, TaskSpec, LMConfig
from transformer_lens import HookedTransformer
from kdfp.schemas.mixins import DtypeSerializerMixin
from kdfp.utils.shallow_model_loading import get_indices_and_num_layers_to_keep
from kdfp.schemas.objective_weights import OBJECTIVE_WEIGHT_VARIANTS_MAP
from kdfp.schemas.subConfigs import (
    BaseDirConfig,
    ExpDirConfig,
    OffProjParamsConfig,
    StudentConfigs,
    LMParamsConfig,
    TrainingParamsConfig,
    HyperparamsConfig,
    TaskConfig,
    DistillationHooks,
    ReductionHooks,
    DistillationDims,
    ReductionDims,
    ProjectionConfig)
from kdfp.utils.enums import (ProjectionDirection, ProjectorTraining, StudentReduction, safe_enum)

class GlobalConfig(ConfigBase):
    teacher_name: str
    students: StudentConfigs
    directories: BaseDirConfig
    offline_projector_params: OffProjParamsConfig
    lm_params: LMParamsConfig
    training_params: TrainingParamsConfig

    hook_types: Dict[str, List[str]] = Field(..., description="Keys 'single' and 'per_layer' with lists of hook names")
    projection_variants: Dict[str, List[str]] = Field(..., description="Keys 'directions' and 'training_methods'")
    objective_weight_variants: List[str]
    jsd_betas: List[float]
    hyperparams: HyperparamsConfig
    tasks: Dict[str, TaskConfig] # A dictionary where the key is the task name (e.g., 'hellaswag') and the value is its config
    hooks: Optional[DistillationHooks] = None
    dims: Optional[DistillationDims] = None

    # validator to ensure hook point keys are expected
    @field_validator('hook_types')
    def validate_hook_keys(cls, v):
        allowed_keys = {'single', 'per_layer'}
        if not all(key in allowed_keys for key in v.keys()):
            raise ValueError(f'hook_types can only contain the keys: {allowed_keys}')
        return v
    
    # validator to parse dtype string into an actual dtype
    @field_validator('torch_dtype')
    def validate_dtype(cls, v):
        return getattr(torch, v)
    
    @model_validator(mode='after')
    def generate_both_hook_lists(self) -> 'GlobalConfig':
        # self.hooks = DistillationHooks()
        tn_hooks, tn_dims = self._generate_hook_list(model_name=self.teacher_name)
        ts_hooks, ts_dims = self._generate_hook_list(model_name=self.teacher_name,
                                                      layer_reduction_factor=self.students.shallow.factor,
                                                      selection_method='linspace')
        """
        - The shallow student is first n layers because once we actually load in the shallow student,
          the first n layers will be the entire student model.
        - This way, the linspaced teacher layers match up with all of the student layers
        """
        sn_hooks, sn_dims = self._generate_hook_list(model_name=self.students.narrow.name)
        ss_hooks, ss_dims = self._generate_hook_list(model_name=self.students.shallow.name,
                                                      layer_reduction_factor=self.students.shallow.factor,
                                                      selection_method='first_n')
        
        teacher_hooks = ReductionHooks(narrow=tn_hooks, shallow=ts_hooks)
        student_hooks = ReductionHooks(narrow=sn_hooks, shallow=ss_hooks)
        self.hooks = DistillationHooks(teacher=teacher_hooks, student=student_hooks)

        teacher_dims = ReductionDims(narrow=tn_dims, shallow=ts_dims)
        student_dims = ReductionDims(narrow=sn_dims, shallow=ss_dims)
        self.dims = DistillationDims(teacher=teacher_dims, student=student_dims)

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
        # binary_tasks = []
        for k, v in self.tasks.items():
            t = TaskSpec(task_name=k, dataset_name=v.dataset_name, subset=v.subset)
            tasks.append(t)
        return tasks
    
    def _generate_hook_list(self, model_name: str,
                            layer_reduction_factor: Optional[float] = None,
                            selection_method: Optional[Literal['linspace', 'first_n']] = None) -> List[str]:
        lm = HookedTransformer.from_pretrained(model_name)
        lm_cfg = lm.cfg
        del lm
        torch.cuda.empty_cache()

        d_model = lm_cfg.d_model
        if layer_reduction_factor is not None:
            assert selection_method, "You must supply a selection method if performing layer reduction"
            layer_idx_list, _ = get_indices_and_num_layers_to_keep(total_layers=lm_cfg.n_layers,
                                                         layer_reduction_factor=layer_reduction_factor,
                                                         selection_method=selection_method)
        else:
            layer_idx_list = list(range(lm_cfg.n_layers))

        hooks = []
        hooks.extend(self.hook_types["single"])
        for idx in layer_idx_list:
            for hp in self.hook_types["per_layer"]:
                hooks.append(f"blocks.{idx}.{hp}")
        return hooks, d_model

    def generate_experiment_list(self, debug: bool = False) -> List[ExperimentConfig]:
        self.tasks = self.generate_task_list()

        base_cache_dir = self.directories.root + '/' + self.directories.cache
        base_model_dir = self.directories.root + '/' + self.directories.models
        base_scores_dir = self.directories.root + '/' + self.directories.scores

        experiment_list = []

        for proj_dir in self.projection_variants["directions"]:
            for proj_train in self.projection_variants["training_methods"]: 
                for obj_weight_variant in self.objective_weight_variants:
                    for jsd_beta in self.jsd_betas:
                            for off_proj_align_beta in self.offline_projector_params.alignment_betas:
                                for reduction in [member for member in StudentReduction]:
                                    if reduction == StudentReduction.NARROW:
                                        proj_str = f"{proj_dir}_{proj_train}"
                                        if proj_train == ProjectorTraining.OFFLINE.value:
                                            proj_str += f"_align_{off_proj_align_beta}"
                                        proj_str += '-'
                                        teacher = LMConfig(model_name=self.teacher_name,
                                                     hook_list=self.hooks.teacher.narrow,
                                                     d_model=self.dims.teacher.narrow)
                                        student = LMConfig(model_name=self.students.narrow.name,
                                                     hook_list=self.hooks.student.narrow,
                                                     d_model=self.dims.student.narrow)
                                        projection_direction = proj_dir
                                        projector_training = proj_train
                                        offline_projector_alignment_beta = off_proj_align_beta


                                    elif reduction == StudentReduction.SHALLOW:
                                        proj_str = ""
                                        teacher = LMConfig(model_name=self.teacher_name,
                                                     hook_list=self.hooks.teacher.shallow,
                                                     d_model=self.dims.teacher.shallow)
                                        student = LMConfig(model_name=self.students.shallow.name,
                                                     hook_list=self.hooks.student.shallow,
                                                     d_model=self.dims.student.shallow)
                                        projection_direction = None
                                        projector_training = None
                                        offline_projector_alignment_beta = None

                                    else:
                                        raise ValueError(f"Invalid reduction specified: '{reduction}'.")


                                    exp_name = f"{reduction.value}-{proj_str}{obj_weight_variant}-jsd_{jsd_beta}"
                                    projection_cfg = ProjectionConfig(
                                        direction=safe_enum(ProjectionDirection, projection_direction),
                                        training=safe_enum(ProjectorTraining, projector_training),
                                        offline_projector_alignment_beta=offline_projector_alignment_beta,
                                        offline_projector_training_args=self.offline_projector_params.training_args,
                                    )

                                    # only use cache for offline projector training
                                    cache_dir = base_cache_dir if projector_training == ProjectorTraining.OFFLINE.value else None
                                    projector_directory = f"{base_model_dir}/offline_projectors/off_proj_{offline_projector_alignment_beta}" if offline_projector_alignment_beta else None
                                    exp_dirs = ExpDirConfig(cache=cache_dir,
                                                            off_proj_model=projector_directory,
                                                            teacher_models=f"{base_model_dir}/teachers/{teacher.model_name}",
                                                            student_model=f"{base_model_dir}/students/{exp_name}",
                                                            student_scores=f"{base_scores_dir}/students/{exp_name}")
                                    exp = ExperimentConfig(
                                        name=exp_name,
                                        dirs=exp_dirs,
                                        reduction=reduction,
                                        teacher=teacher,
                                        student_cfg=student,
                                        training_params=self.training_params,
                                        lm_params=self.lm_params,
                                        projection=projection_cfg,
                                        objective_weight_variant=obj_weight_variant,
                                        objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP[obj_weight_variant],
                                        jsd_beta=jsd_beta,
                                        hyperparams=self.hyperparams,
                                        tasks=self.tasks,
                                        torch_dtype=self.torch_dtype,
                                    )
                                    experiment_list.append(exp)

        # need to add in baseline experiments
        SHALLOW_FT_NAME = "shallow_ft"
        ft_shallow_student = ExperimentConfig(
            name=SHALLOW_FT_NAME,
            dirs=ExpDirConfig(cache=None,
                              teacher_models=f"{base_model_dir}/{self.teacher_name}",
                              student_model=f"{base_model_dir}/{SHALLOW_FT_NAME}",
                              student_scores=f"{base_scores_dir}/{SHALLOW_FT_NAME}"),
            reduction=StudentReduction.SHALLOW,
            teacher=LMConfig(model_name=self.teacher_name,
                       d_model=self.dims.teacher.shallow),
            student_cfg=LMConfig(model_name=self.students.shallow.name,
                       layer_reduction_factor=self.students.shallow.factor,
                       d_model=self.dims.student.shallow),
            training_params=self.training_params,
            lm_params=self.lm_params,
            projection=None,
            objective_weight_variant="finetuning",
            objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP["finetuning"],
            jsd_beta=None,
            hyperparams=self.hyperparams,
            tasks=self.tasks,
            torch_dtype=self.torch_dtype,
        )
        NARROW_FT_NAME = "narrow_ft"
        ft_narrow_student = ExperimentConfig(
            name=NARROW_FT_NAME,
            dirs=ExpDirConfig(cache=None,
                              teacher_models=f"{base_model_dir}/{self.teacher_name}",
                              student_model=f"{base_model_dir}/{NARROW_FT_NAME}",
                              student_scores=f"{base_scores_dir}/{NARROW_FT_NAME}"),
            reduction=StudentReduction.NARROW,
            teacher=LMConfig(model_name=self.teacher_name,
                       d_model=self.dims.teacher.narrow),
            student_cfg=LMConfig(model_name=self.students.narrow.name,
                       d_model=self.dims.student.narrow),
            training_params=self.training_params,
            lm_params=self.lm_params,
            projection=None,
            objective_weight_variant="finetuning",
            objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP["finetuning"],
            jsd_beta=None,
            hyperparams=self.hyperparams,
            tasks=self.tasks,
            torch_dtype=self.torch_dtype,
        )

        experiment_list.extend([ft_shallow_student, ft_narrow_student])

        if debug:
            print(f"Total experiments created: {len(experiment_list)}")

        # NOTE: since I didn't want to build the logic into the loop, duplicate experiments will be created
        # - ConfigBase provides the methods to allow me to hash and compare experiments
        # - I use this to filter out the duplicates
        unique_experiment_list = list(set(experiment_list))
        exp_names = [e.name for e in unique_experiment_list]
        self.sorted_experiment_list = sorted(unique_experiment_list, key=lambda exp: exp.name)

        return self.sorted_experiment_list, sorted(exp_names)

    def run_baseline_tests(self) -> List[dict]:
        results_list = []
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            deterministic=self.training_params.reproducible_backend,
        )
        for task in self.tasks:
            base_teacher_cfg = BaselineModuleConfig(
                model_name=self.teacher_name,
                is_shallow=False,
                task=task,
                seq_len=self.training_params.seq_len,
                batch_size=self.lm_params.test_batch_size,
                seed=self.training_params.seed,
                torch_dtype=self.training_params.torch_dtype,
                num_data_workers=self.training_params.resources.teacher.num_data_workers,
                n_samples=self.lm_params.num_samples,
                layer_reduction_factor=None,)
            base_teacher_module = BaselineModule(base_teacher_cfg)
            base_teacher_results = trainer.test(base_teacher_module)
            results_list.append({
                "EXPERIMENT": "base_teacher",
                "TASK": task.task_name,
                "METRIC": "accuracy",
                "SCORE": base_teacher_results[0]['test_acc']})
            results_list.append({
                "EXPERIMENT": "base_teacher",
                "TASK": task.task_name,
                "METRIC": "f1",
                "SCORE": base_teacher_results[0]['test_f1']})

            base_narrow_student_cfg = BaselineModuleConfig(
                model_name=self.students.narrow.name,
                is_shallow=False,
                task=task,
                seq_len=self.training_params.seq_len,
                batch_size=self.lm_params.test_batch_size,
                seed=self.training_params.seed,
                torch_dtype=self.training_params.torch_dtype,
                num_data_workers=self.training_params.resources.teacher.num_data_workers,
                n_samples=self.lm_params.num_samples,
                layer_reduction_factor=None,
            )
            base_narrow_student_module = BaselineModule(base_narrow_student_cfg)
            base_narrow_student_results = trainer.test(base_narrow_student_module)
            results_list.append({
                "EXPERIMENT": "base_narrow",
                "TASK": task.task_name,
                "METRIC": "accuracy",
                "SCORE": base_narrow_student_results[0]['test_acc']})
            results_list.append({
                "EXPERIMENT": "base_teacher",
                "TASK": task.task_name,
                "METRIC": "f1",
                "SCORE": base_narrow_student_results[0]['test_f1']})

            base_shallow_student_cfg = BaselineModuleConfig(
                model_name=self.students.shallow.name,
                is_shallow=True,
                task=task,
                seq_len=self.training_params.seq_len,
                batch_size=self.lm_params.test_batch_size,
                seed=self.training_params.seed,
                torch_dtype=self.training_params.torch_dtype,
                num_data_workers=self.training_params.resources.teacher.num_data_workers,
                n_samples=self.lm_params.num_samples,
                layer_reduction_factor=self.students.shallow.factor,
            )
            base_shallow_student_module = BaselineModule(base_shallow_student_cfg)
            base_shallow_student_results = trainer.test(base_shallow_student_module)
            results_list.append({
                "EXPERIMENT": "base_shallow",
                "TASK": task.task_name,
                "METRIC": "accuracy",
                "SCORE": base_shallow_student_results[0]['test_acc']})
            results_list.append({
                "EXPERIMENT": "base_teacher",
                "TASK": task.task_name,
                "METRIC": "f1",
                "SCORE": base_shallow_student_results[0]['test_f1']})
        return results_list
