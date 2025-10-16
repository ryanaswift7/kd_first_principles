from typing import Optional, Literal, List, Dict, Any, Union
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM
from kdfp.core.projector.offline.offlineProjector import gen_off_proj_weights_filename
from kdfp.core.projector.randomized import RandomizedProjector
from kdfp.core.student.student import StudentModule
from kdfp.core.teacher import TeacherModule, gen_teacher_weights_filename
from kdfp.schemas.mixins import DtypeSerializerMixin
from kdfp.schemas.objective_weights import ObjectiveWeights
from kdfp.schemas.tasks import TaskSpec
from kdfp.schemas.lmConfig import LMConfig
from kdfp.schemas.configBase import ConfigBase
from pathlib import Path
import json
from pydantic import model_validator
from kdfp.schemas.subConfigs import (
    ExpDirConfig,
    LMParamsConfig, 
    OffProjTrainArgsConfig,
    TrainingParamsConfig,
    HyperparamsConfig,
    ProjectionConfig)
from kdfp.core.projector import OfflineProjectorModule, OnlineProjector, OFF_PROJ_STATE_DICT_FILE
from kdfp.utils.activation_caching import activation_caching
import torch.nn as nn
from kdfp.utils.enums import (StudentReduction, ProjectorTraining)
import torch
import pandas as pd

class ExperimentConfig(ConfigBase, DtypeSerializerMixin):
    name: str
    reduction: StudentReduction
    teacher_cfg: LMConfig
    student_cfg: LMConfig
    dirs: ExpDirConfig
    training_params: TrainingParamsConfig
    lm_params: LMParamsConfig
    projection: Optional[ProjectionConfig]
    objective_weight_variant: str
    objective_weights: ObjectiveWeights
    jsd_beta: Optional[float]
    hyperparams: HyperparamsConfig
    tasks: List[TaskSpec]

    @model_validator(mode='after')
    def validate_and_transform_config(self) -> 'ExperimentConfig':
        if self.projection:
            if self.reduction == StudentReduction.SHALLOW:
                # all of the fields are already none but I'd prefer
                # to also make the whole object None
                self.projection = None

            elif self.projection.training != ProjectorTraining.OFFLINE:
                self.projection.offline_projector_alignment_beta = None
                self.projection.offline_projector_training_args = None
        return self

    def pretty_print_config(self) -> None:
        """Pretty-prints the complete configuration using Pydantic's JSON serialization."""
        print(self.model_dump_json(indent=4))

    def save_config_json(self, save_dir: Union[str, Path]) -> Path:
        """
        Saves the complete configuration to a 'config.json' file.

        Args:
            save_dir: The directory where the config file will be saved.

        Returns:
            The path to the saved JSON file.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / 'config.json'

        # Pydantic's model_dump_json handles the entire serialization process,
        # including nested models, Path objects, enums, etc.
        file_path.write_text(self.model_dump_json(indent=4))
        
        print(f"Configuration saved to: {file_path}")
        return file_path

    # only offline projectors are initialized here bc don't want to pass in models to
    # lightning modules, so I load them in the student module 
    def _init_offline_projector(self,
                                # cache_dir: str,
                                # save_dir: str,
                                teacher_dir: Optional[str] = None,
                                student_dir: Optional[str] = None,) -> OfflineProjectorModule:
        """Initialize the offline projector for an experiment. Defaults to pretrained teacher
            and student, but you may use a custom version instead by passing in the directories.
        """
        # if we already trained one, load it instead of creating another
        off_proj_weights_path = Path(self.dirs.off_proj_model) / gen_off_proj_weights_filename(alignment_beta=self.projection.offline_projector_alignment_beta)
        if off_proj_weights_path.exists():
            print(f"Pretrained offline projector for with alignment beta '{self.projection.offline_projector_alignment_beta}' located at: {str(off_proj_weights_path)}")

            # p = OfflineProjectorModule.load_pretrained(self.dirs.off_proj_model)
            # return p.extract_single_direction(direction=self.projection.direction)
        else:
            # cache teacher and student acts
            teacher_cache_dir, teacher_dim = activation_caching(lm_name=self.teacher_cfg.model_name,
                                                    lm_hook_points=self.teacher_cfg.hook_list,
                                                    dataset_name=self.projection.offline_projector_training_args.dataset.name,
                                                    dataset_subset=self.projection.offline_projector_training_args.dataset.subset,
                                                    split=self.projection.offline_projector_training_args.dataset.split,
                                                    seq_len=self.training_params.seq_len,
                                                    batch_size=self.projection.offline_projector_training_args.cache_batch_size,
                                                    cache_dir=self.dirs.cache,
                                                    token_rename=self.projection.offline_projector_training_args.dataset.token_rename,
                                                    shuffle=self.projection.offline_projector_training_args.dataset.shuffle,
                                                    lm_dir=teacher_dir,
                                                    total_tokens=self.projection.offline_projector_training_args.total_tokens,
                                                    force_cache_all=self.projection.offline_projector_training_args.force_cache_all,
                                                    cache_outputs=False,
                                                    num_workers=self.training_params.num_data_workers,
                                                    device=self.training_params.device,
                                                    torch_dtype=self.training_params.torch_dtype,)
            
            student_cache_dir, student_dim = activation_caching(lm_name=self.student_cfg.model_name,
                                                    lm_hook_points=self.student_cfg.hook_list,
                                                    dataset_name=self.projection.offline_projector_training_args.dataset.name,
                                                    dataset_subset=self.projection.offline_projector_training_args.dataset.subset,
                                                    split=self.projection.offline_projector_training_args.dataset.split,
                                                    seq_len=self.training_params.seq_len,
                                                    batch_size=self.projection.offline_projector_training_args.cache_batch_size,
                                                    cache_dir=self.dirs.cache,
                                                    token_rename=self.projection.offline_projector_training_args.dataset.token_rename,
                                                    shuffle=self.projection.offline_projector_training_args.dataset.shuffle,
                                                    lm_dir=student_dir,
                                                    total_tokens=self.projection.offline_projector_training_args.total_tokens,
                                                    force_cache_all=self.projection.offline_projector_training_args.force_cache_all,
                                                    cache_outputs=False,
                                                    num_workers=self.training_params.resources.projector.num_data_workers,
                                                    device=self.training_params.device,
                                                    torch_dtype=self.training_params.torch_dtype,)

            # create projector
            projector = OfflineProjectorModule.create(
                teacher_cache_dir=teacher_cache_dir,
                student_cache_dir=student_cache_dir,
                teacher_dim=teacher_dim,
                student_dim=student_dim,
                projection=self.projection,
                hyperparams=self.hyperparams,
                training_params=self.training_params,
                val_split=0.1,
                test_split=0.1,)
            projector.save_pretrained(path=self.dirs.off_proj_model)
            # load_test = OfflineProjector.load_pretrained(save_path=self.dirs.off_proj_model)
            # return projector.extract_single_direction(direction=self.projection.direction)

    def init_online_projector(self):
        return OnlineProjector(teacher_dim=self.teacher_cfg.d_model,
                            student_dim=self.student_cfg.d_model,
                            direction=self.projection.direction,
                            torch_dtype=self.torch_dtype)
    def init_randomized_projector(self):
        return RandomizedProjector(teacher_dim=self.teacher_cfg.d_model,
                            student_dim=self.student_cfg.d_model,
                            direction=self.projection.direction,
                            torch_dtype=self.torch_dtype)

    def init_projector(self) -> None:
        """Initialize projector for an experiment. kwargs are for init_offline_projector."""
        if self.projection:
            if self.projection.training == ProjectorTraining.ONLINE:
                # self.init_online_projector()
                print(f"{self.projection.training.value.upper()} projector will be initalized during the experiment.")
            
            elif self.projection.training == ProjectorTraining.OFFLINE:
                print(f"{self.projection.training.value.upper()} projector initializtion starting...")
                self._init_offline_projector()
                print(f"{self.projection.training.value.upper()} projector initializtion complete!")

            elif self.projection.training == ProjectorTraining.RANDOMIZED:
                # self.init_randomized_projector()
                print(f"{self.projection.training.value.upper()} projector will be initalized during the experiment.")

            else:
                raise ValueError(f"Invalid projection training specified: {self.projection.training}.")
            
        return

    def init_teacher_by_task(self, task: TaskSpec) -> None:
        # save_dir = f"{self.dirs.teacher_model}/{task.task_name}"
        teacher_weights_file = Path(self.dirs.teacher_models) / gen_teacher_weights_filename(task.task_name)
        if teacher_weights_file.exists():
            print(f"Pretrained teacher for task '{task.task_name}' located at: {str(teacher_weights_file)}")
        else:
            t = TeacherModule.create(
                lm_name=self.teacher_cfg.model_name,
                task=task,
                lm_params=self.lm_params,
                hyperparams=self.hyperparams,
                training_params=self.training_params,
            )
            t.save_pretrained(path=self.dirs.teacher_models)


    def init(self) -> None:
        """Initialize teachers, student, and projector"""
        self.init_projector()
        for task in self.tasks:
            self.init_teacher_by_task(task)

    def run(self) -> pd.Dataframe:
        results_list = []
        test_trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            deterministic=self.training_params.reproducible_backend,
        )
        for task in self.tasks:
            student = StudentModule.create(
                exp_name=self.name,
                dirs=self.dirs,
                student_cfg=self.student_cfg,
                teacher_cfg=self.teacher_cfg,
                task=task,
                lm_params=self.lm_params,
                hyperparams=self.hyperparams,
                training_params=self.training_params,
                projection=self.projection,
                jsd_beta=self.jsd_beta,
                objective_weights=self.objective_weights)
            results = test_trainer.test(student)
            results_list.append({
                "EXPERIMENT": self.name,
                "TASK": task.task_name,
                "METRIC": "accuracy",
                "SCORE": results[0]['test_acc']})
            results_list.append({
                "EXPERIMENT": self.name,
                "TASK": task.task_name,
                "METRIC": "f1",
                "SCORE": results[0]['test_f1']})
        return pd.DataFrame(results_list)
            
