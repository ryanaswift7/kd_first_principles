from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import Field, computed_field, field_validator, model_validator
from transformer_lens import HookedTransformer

from kdfp.core.baseline import BaselineModule, BaselineModuleConfig
from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.directories import BaseDirConfig
from kdfp.schemas.experimentConfig import ExperimentConfig, LMConfig, TaskInstance
from kdfp.schemas.hparams import HparamsConfig
from kdfp.schemas.models import (
    DistillationGroups,
    DistillationModel,
    DistillationPair,
    ModelSelection,
)
from kdfp.schemas.objective_weights import OBJECTIVE_WEIGHT_VARIANTS_MAP
from kdfp.schemas.projections import ProjectionConfig
from kdfp.schemas.tasks import TaskConfig
from kdfp.schemas.trainingParams import TrainingParamsConfig
from kdfp.utils.enums import (
    ProjectionDirection,
    ProjectorTraining,
    StudentReduction,
    safe_enum,
)
from kdfp.utils.general import get_hook_block_idx, get_timestamp
from kdfp.utils.results import generate_result
from kdfp.utils.shallow_model_loading import get_indices_and_num_layers_to_keep


class SuiteConfig(ConfigBase):
    suite_title: str
    models: ModelSelection
    directories: BaseDirConfig
    training_params: TrainingParamsConfig
    hook_types: Dict[str, List[str]] = Field(
        ..., description="Keys 'single' and 'per_layer' with lists of hook names"
    )
    projection: ProjectionConfig
    hparams: HparamsConfig
    # A dictionary where the key is the task name
    # (e.g., 'hellaswag') and the value is its config
    tasks: Dict[str, TaskConfig]

    @computed_field
    @property
    def task_specs(self) -> List[TaskInstance]:
        return [
            TaskInstance(name=name, dataset_name=cfg.dataset_name, subset=cfg.subset)
            for name, cfg in self.tasks.items()
        ]

    # validator to ensure hook point keys are expected
    @field_validator("hook_types")
    def validate_hook_keys(cls, v):
        allowed_keys = {"single", "per_layer"}
        if not all(key in allowed_keys for key in v.keys()):
            raise ValueError(f"hook_types can only contain the keys: {allowed_keys}")
        return v

    @model_validator(mode="after")
    def build_distill_groups(self) -> "SuiteConfig":
        raw_teacher_pretrained: DistillationModel = self._generate_hook_list(
            model_name=self.models.teacher.name,
        )

        student_pretrained: DistillationModel = self._generate_hook_list(
            model_name=self.models.pretrained.name,
        )

        teacher_shallow: DistillationModel = self._generate_hook_list(
            model_name=self.models.teacher.name,
            layer_reduction_factor=self.models.shallow.layer_reduction_factor,
            selection_method="linspace",
        )

        """
        - The shallow student is first n layers because once we actually load in the shallow student,
          the first n layers will be the entire student model.
        - This way, the linspaced teacher layers match up with all of the student layers
        - NOTE: for narrow students, we initialize the full hook list here and let the
                ExperimentConfig apply the kd_layer_filter_factors
        """
        student_shallow: DistillationModel = self._generate_hook_list(
            model_name=self.models.shallow.name,
            layer_reduction_factor=self.models.shallow.layer_reduction_factor,
            selection_method="first_n",
        )

        assert raw_teacher_pretrained.d_model == teacher_shallow.d_model

        # need to select teacher blocks which align with the pretrained student's
        def _filter_teacher_hooks_for_pretrained_student(
            raw_teacher_pretrained: DistillationModel,
            student_pretrained: DistillationModel,
        ) -> DistillationModel:
            # compute block indices to keep
            # robust to student layers <= teacher layers
            # ensures no collisions in the block indices (i.e. full distillation across all pretrained student layers)
            teacher_n_distilled_layers = min(
                raw_teacher_pretrained.n_distilled_layers,
                student_pretrained.n_distilled_layers,
            )
            selected_blocks = np.arange(teacher_n_distilled_layers)
            selected_blocks = (
                selected_blocks * (raw_teacher_pretrained.n_distilled_layers - 1)
            ) // (teacher_n_distilled_layers - 1)
            assert len(set(selected_blocks)) == len(selected_blocks)
            selected_blocks = set(selected_blocks)

            filtered_teacher_hooks = []
            for hook in raw_teacher_pretrained.hook_list:
                # add in all hooks outside of transformer blocks
                if "blocks" not in hook:
                    filtered_teacher_hooks.append(hook)
                else:
                    block_idx = get_hook_block_idx(hook)
                    if block_idx in selected_blocks:
                        filtered_teacher_hooks.append(hook)

            corrected_teacher_pretrained = deepcopy(raw_teacher_pretrained)
            corrected_teacher_pretrained.hook_list = filtered_teacher_hooks
            corrected_teacher_pretrained.n_distilled_layers = teacher_n_distilled_layers

            return corrected_teacher_pretrained

        corrected_teacher_pretrained = _filter_teacher_hooks_for_pretrained_student(
            raw_teacher_pretrained, student_pretrained
        )

        shallow_pair = DistillationPair(
            teacher=teacher_shallow, student=student_shallow
        )
        pretrained_pair = DistillationPair(
            teacher=corrected_teacher_pretrained,
            student=student_pretrained,
        )

        self.distill_groups = DistillationGroups(
            shallow=shallow_pair,
            pretrained=pretrained_pair,
        )

        return self

    # Class method to load from a YAML file
    @classmethod
    def from_yaml(cls, file_path: str) -> "SuiteConfig":
        with open(file_path, "r") as f:
            raw_config = yaml.safe_load(f)
        return cls(**raw_config)

    # Class method to load from a YAML string (useful for testing)
    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> "SuiteConfig":
        raw_config = yaml.safe_load(yaml_str)
        return cls(**raw_config)

    def _generate_hook_list(
        self,
        model_name: str,
        layer_reduction_factor: Optional[float] = None,
        selection_method: Optional[Literal["linspace", "first_n"]] = None,
    ) -> DistillationModel:
        lm = HookedTransformer.from_pretrained(model_name)
        lm_cfg = lm.cfg
        del lm
        torch.cuda.empty_cache()

        d_model = lm_cfg.d_model
        if layer_reduction_factor is not None:
            assert selection_method, (
                "You must supply a selection method if performing layer reduction"
            )
            layer_idx_list, n_layers_to_keep = get_indices_and_num_layers_to_keep(
                total_layers=lm_cfg.n_layers,
                layer_reduction_factor=layer_reduction_factor,
                selection_method=selection_method,
            )
        else:
            layer_idx_list = list(range(lm_cfg.n_layers))
            n_layers_to_keep = lm_cfg.n_layers

        hooks = []
        hooks.extend(self.hook_types["single"])
        for idx in layer_idx_list:
            for hp in self.hook_types["per_layer"]:
                hooks.append(f"blocks.{idx}.{hp}")
        return DistillationModel(
            name=model_name,
            hook_list=hooks,
            d_model=d_model,
            n_distilled_layers=n_layers_to_keep,
            layer_reduction_factor=layer_reduction_factor,
        )

    def generate_experiment_list(self, debug: bool = False) -> List[ExperimentConfig]:
        base_cache_dir = self.directories.root + "/" + self.directories.cache
        base_model_dir = self.directories.root + "/" + self.directories.models
        base_scores_dir = self.directories.root + "/" + self.directories.scores
        ray_dir = self.directories.root + "/" + self.directories.ray
        lightning_dir = self.directories.root + "/" + self.directories.lightning

        experiment_list = []

        for proj_dir in self.projection_variants["directions"]:
            for proj_train in self.projection_variants["training_methods"]:
                for obj_weight_variant in self.objective_weight_variants:
                    for jsd_beta in self.jsd_betas:
                        for (
                            off_proj_align_beta
                        ) in self.offline_projector_params.alignment_betas:
                            for reduction in [member for member in StudentReduction]:
                                for (
                                    kdlff
                                ) in self.students.narrow.kd_layer_filter_factors:
                                    if reduction == StudentReduction.NARROW:
                                        filter_str = f"-filter_{kdlff:.3f}"
                                        proj_str = f"{proj_dir}_{proj_train}"
                                        if (
                                            proj_train
                                            == ProjectorTraining.OFFLINE.value
                                        ):
                                            proj_str += f"_align_{off_proj_align_beta}"
                                        proj_str += "-"
                                        teacher_cfg = LMConfig(
                                            model_name=self.teacher_name,
                                            hook_list=self.hooks.teacher.narrow,
                                            d_model=self.dims.teacher.narrow,
                                            kd_layer_filter_factor=kdlff,
                                            n_layers=self.layers.teacher_narrow,
                                        )
                                        student_cfg = LMConfig(
                                            model_name=self.students.narrow.name,
                                            hook_list=self.hooks.student.narrow,
                                            d_model=self.dims.student.narrow,
                                            kd_layer_filter_factor=kdlff,
                                            n_layers=self.layers.student_narrow,
                                        )
                                        projection_direction = proj_dir
                                        projector_training = proj_train
                                        offline_projector_alignment_beta = (
                                            off_proj_align_beta
                                        )

                                    elif reduction == StudentReduction.SHALLOW:
                                        filter_str = ""
                                        proj_str = ""
                                        teacher_cfg = LMConfig(
                                            model_name=self.teacher_name,
                                            hook_list=self.hooks.teacher.shallow,
                                            d_model=self.dims.teacher.shallow,
                                            n_layers=self.layers.teacher_shallow,
                                        )
                                        student_cfg = LMConfig(
                                            model_name=self.students.shallow.name,
                                            layer_reduction_factor=self.students.shallow.layer_reduction_factor,
                                            hook_list=self.hooks.student.shallow,
                                            d_model=self.dims.student.shallow,
                                            n_layers=self.layers.student_shallow,
                                        )
                                        projection_direction = None
                                        projector_training = None
                                        offline_projector_alignment_beta = None

                                    else:
                                        raise ValueError(
                                            f"Invalid reduction specified: '{reduction}'."
                                        )

                                    exp_name = f"{reduction.value}-{proj_str}{obj_weight_variant}-jsd_{jsd_beta}{filter_str}"
                                    projection_cfg = ProjectionConfig(
                                        direction=safe_enum(
                                            ProjectionDirection, projection_direction
                                        ),
                                        training=safe_enum(
                                            ProjectorTraining, projector_training
                                        ),
                                        offline_projector_alignment_beta=offline_projector_alignment_beta,
                                        offline_projector_training_args=self.offline_projector_params.training_args,
                                    )

                                    # only use cache for offline projector training
                                    cache_dir = (
                                        base_cache_dir
                                        if projector_training
                                        == ProjectorTraining.OFFLINE.value
                                        else None
                                    )
                                    projector_directory = (
                                        f"{base_model_dir}/offline_projectors/off_proj_{offline_projector_alignment_beta}"
                                        if offline_projector_alignment_beta is not None
                                        else None
                                    )
                                    exp_dirs = ExpDirConfig(
                                        cache=cache_dir,
                                        off_proj_models=projector_directory,
                                        teacher_model=f"{base_model_dir}/teachers/{teacher_cfg.model_name}",
                                        teacher_scores=f"{base_scores_dir}/teachers/{teacher_cfg.model_name}",
                                        student_model=f"{base_model_dir}/students/{exp_name}",
                                        student_scores=f"{base_scores_dir}/students/{exp_name}",
                                        ray=ray_dir,
                                        lightning=lightning_dir,
                                    )

                                    exp = ExperimentConfig(
                                        name=exp_name,
                                        dirs=exp_dirs,
                                        reduction=reduction,
                                        teacher_cfg=teacher_cfg,
                                        student_cfg=student_cfg,
                                        training_params=self.training_params,
                                        lm_params=self.lm_params,
                                        projection=projection_cfg,
                                        objective_weight_variant=obj_weight_variant,
                                        objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP[
                                            obj_weight_variant
                                        ],
                                        jsd_beta=jsd_beta,
                                        hparams=self.hyperparams,
                                        tasks=self.tasks,
                                    )
                                    experiment_list.append(exp)

        if self.training_params.get_baselines:
            # need to add in baseline experiments
            SHALLOW_FT_NAME = "shallow_ft"
            ft_shallow_student = ExperimentConfig(
                name=SHALLOW_FT_NAME,
                dirs=ExpDirConfig(
                    cache=None,
                    off_proj_models=None,
                    teacher_model=f"{base_model_dir}/teachers/{self.teacher_name}",
                    teacher_scores=f"{base_scores_dir}/teachers/{self.teacher_name}",
                    student_model=f"{base_model_dir}/students/{SHALLOW_FT_NAME}",
                    student_scores=f"{base_scores_dir}/students/{SHALLOW_FT_NAME}",
                    ray=ray_dir,
                    lightning=lightning_dir,
                ),
                reduction=StudentReduction.SHALLOW,
                teacher_cfg=LMConfig(
                    model_name=self.teacher_name,
                    d_model=self.dims.teacher.shallow,
                    n_layers=self.layers.teacher_shallow,
                ),
                student_cfg=LMConfig(
                    model_name=self.students.shallow.name,
                    layer_reduction_factor=self.students.shallow.layer_reduction_factor,
                    d_model=self.dims.student.shallow,
                    n_layers=self.layers.student_shallow,
                ),
                training_params=self.training_params,
                lm_params=self.lm_params,
                projection=None,
                objective_weight_variant="finetuning",
                objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP["finetuning"],
                jsd_beta=None,
                hparams=self.hyperparams,
                tasks=self.tasks,
            )
            NARROW_FT_NAME = "narrow_ft"
            ft_narrow_student = ExperimentConfig(
                name=NARROW_FT_NAME,
                dirs=ExpDirConfig(
                    cache=None,
                    off_proj_models=None,
                    teacher_model=f"{base_model_dir}/teachers/{self.teacher_name}",
                    teacher_scores=f"{base_scores_dir}/teachers/{self.teacher_name}",
                    student_model=f"{base_model_dir}/students/{NARROW_FT_NAME}",
                    student_scores=f"{base_scores_dir}/students/{NARROW_FT_NAME}",
                    ray=ray_dir,
                    lightning=lightning_dir,
                ),
                reduction=StudentReduction.NARROW,
                teacher_cfg=LMConfig(
                    model_name=self.teacher_name,
                    d_model=self.dims.teacher.narrow,
                    n_layers=self.layers.teacher_narrow,
                ),
                student_cfg=LMConfig(
                    model_name=self.students.narrow.name,
                    d_model=self.dims.student.narrow,
                    n_layers=self.layers.student_narrow,
                ),
                training_params=self.training_params,
                lm_params=self.lm_params,
                projection=None,
                objective_weight_variant="finetuning",
                objective_weights=OBJECTIVE_WEIGHT_VARIANTS_MAP["finetuning"],
                jsd_beta=None,
                hparams=self.hyperparams,
                tasks=self.tasks,
            )

            experiment_list.extend([ft_shallow_student, ft_narrow_student])

        if debug:
            print(f"Total experiments created: {len(experiment_list)}")

        # NOTE: since I didn't want to build the logic into the loop, duplicate experiments will be created
        # - ConfigBase provides the methods to allow me to hash and compare experiments
        # - I use this to filter out the duplicates
        unique_experiment_list = list(set(experiment_list))
        exp_names = [e.name for e in unique_experiment_list]
        sorted_experiment_list = sorted(
            unique_experiment_list, key=lambda exp: exp.name
        )

        return sorted_experiment_list, sorted(exp_names)

    def run_baseline_tests(self) -> List[dict]:
        pl.seed_everything(seed=self.training_params.seed, workers=True)
        results_list = []

        def trainer_factory() -> pl.Trainer:
            return pl.Trainer(
                accelerator="gpu",
                strategy="auto",
                devices=1,
                deterministic=self.training_params.reproducible_backend,
                default_root_dir=self.directories.root
                + "/"
                + self.directories.lightning,
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
                layer_reduction_factor=None,
            )
            base_teacher_module = BaselineModule(base_teacher_cfg)
            base_teacher_module.print_param_count("base_teacher")
            base_teacher_results = trainer_factory().test(
                model=base_teacher_module, verbose=True
            )
            results_list.append(
                generate_result(
                    name="base_teacher",
                    model=base_teacher_module.model,
                    task_name=task.name,
                    metric_name="accuracy",
                    score=base_teacher_results[0]["test_acc"],
                )
            )

            base_narrow_student_cfg = BaselineModuleConfig(
                model_name=self.students.narrow.name,
                is_shallow=False,
                task=task,
                seq_len=self.training_params.seq_len,
                batch_size=self.lm_params.test_batch_size,
                seed=self.training_params.seed,
                torch_dtype=self.training_params.torch_dtype,
                num_data_workers=self.training_params.resources.student.num_data_workers,
                n_samples=self.lm_params.num_samples,
                layer_reduction_factor=None,
            )
            base_narrow_student_module = BaselineModule(base_narrow_student_cfg)
            base_narrow_student_module.print_param_count("base_narrow")
            base_narrow_student_results = trainer_factory().test(
                base_narrow_student_module
            )
            results_list.append(
                generate_result(
                    name="base_narrow",
                    model=base_narrow_student_module.model,
                    task_name=task.name,
                    metric_name="accuracy",
                    score=base_narrow_student_results[0]["test_acc"],
                )
            )

            base_shallow_student_cfg = BaselineModuleConfig(
                model_name=self.students.shallow.name,
                is_shallow=True,
                task=task,
                seq_len=self.training_params.seq_len,
                batch_size=self.lm_params.test_batch_size,
                seed=self.training_params.seed,
                torch_dtype=self.training_params.torch_dtype,
                num_data_workers=self.training_params.resources.student.num_data_workers,
                n_samples=self.lm_params.num_samples,
                layer_reduction_factor=self.students.shallow.layer_reduction_factor,
            )
            base_shallow_student_module = BaselineModule(base_shallow_student_cfg)
            base_shallow_student_module.print_param_count("base_shallow")
            base_shallow_student_results = trainer_factory().test(
                base_shallow_student_module
            )
            results_list.append(
                generate_result(
                    name="base_shallow",
                    model=base_shallow_student_module.model,
                    task_name=task.name,
                    metric_name="accuracy",
                    score=base_shallow_student_results[0]["test_acc"],
                )
            )

            scores_dir = (
                Path(self.directories.root) / self.directories.scores / "baselines"
            )
            scores_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results_list).to_parquet(
                scores_dir / f"results_df_{get_timestamp()}.parquet"
            )

        return results_list
