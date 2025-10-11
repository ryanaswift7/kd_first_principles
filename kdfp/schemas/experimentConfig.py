from typing import Optional, Literal, List, Dict, Any, Union

from transformers import AutoModelForCausalLM
from kdfp.core.projector.randomized import RandomizedProjector
from kdfp.core.teacher import Teacher, gen_teacher_weights_file
from kdfp.schemas.objective_weights import ObjectiveWeights
from kdfp.schemas.tasks import TaskSpec
from kdfp.schemas.lm import LM
from kdfp.schemas.configBase import ConfigBase
from pathlib import Path
import json
from pydantic import model_validator
from kdfp.schemas.subConfigs import (
    DtypeSerializerMixin,
    ExpDirConfig,
    LMParamsConfig, 
    OffProjTrainArgsConfig,
    TrainingParamsConfig,
    HyperparamsConfig,
    ExpProjectionConfig)
from kdfp.core.projector import OfflineProjector, OnlineProjector, OFF_PROJ_STATE_DICT_FILE
from kdfp.utils.activation_caching import activation_caching
import torch.nn as nn
from kdfp.utils.enums import (StudentReduction, ProjectorTraining)
import torch

class ExperimentConfig(ConfigBase, DtypeSerializerMixin):
    name: str
    reduction: StudentReduction
    teacher: LM
    student: LM
    torch_dtype: torch.dtype
    dirs: ExpDirConfig
    train_params: TrainingParamsConfig
    lm_params: LMParamsConfig
    projection: Optional[ExpProjectionConfig]
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

    def _init_offline_projector(self,
                                # cache_dir: str,
                                # save_dir: str,
                                teacher_dir: Optional[str] = None,
                                student_dir: Optional[str] = None,) -> OfflineProjector:
        """Initialize the offline projector for an experiment. Defaults to pretrained teacher
            and student, but you may use a custom version instead by passing in the directories.
        """
        # if we already trained one, load it instead of creating another
        weights_path = Path(self.dirs.off_proj_model) / OFF_PROJ_STATE_DICT_FILE
        if weights_path.exists():
            print(f"Loading pretrained projector at: {str(weights_path)}")
            p = OfflineProjector.load_pretrained(self.dirs.off_proj_model)
            return p.extract_single_direction(direction=self.projection.direction)

        # cache teacher and student acts
        teacher_cache_dir, teacher_dim = activation_caching(lm_name=self.teacher.model_name,
                                                lm_hook_points=self.teacher.hook_list,
                                                dataset_name=self.projection.offline_projector_training_args.dataset.name,
                                                dataset_subset=self.projection.offline_projector_training_args.dataset.subset,
                                                split=self.projection.offline_projector_training_args.dataset.split,
                                                seq_len=self.train_params.seq_len,
                                                batch_size=self.projection.offline_projector_training_args.cache_batch_size,
                                                cache_dir=self.dirs.cache,
                                                token_rename=self.projection.offline_projector_training_args.dataset.token_rename,
                                                shuffle=self.projection.offline_projector_training_args.dataset.shuffle,
                                                lm_dir=teacher_dir,
                                                total_tokens=self.projection.offline_projector_training_args.total_tokens,
                                                force_cache_all=self.projection.offline_projector_training_args.force_cache_all,
                                                cache_outputs=False,
                                                num_workers=self.train_params.num_data_workers,
                                                device=self.train_params.device,
                                                torch_dtype=self.torch_dtype,)
        
        student_cache_dir, student_dim = activation_caching(lm_name=self.student.model_name,
                                                lm_hook_points=self.student.hook_list,
                                                dataset_name=self.projection.offline_projector_training_args.dataset.name,
                                                dataset_subset=self.projection.offline_projector_training_args.dataset.subset,
                                                split=self.projection.offline_projector_training_args.dataset.split,
                                                seq_len=self.train_params.seq_len,
                                                batch_size=self.projection.offline_projector_training_args.cache_batch_size,
                                                cache_dir=self.dirs.cache,
                                                token_rename=self.projection.offline_projector_training_args.dataset.token_rename,
                                                shuffle=self.projection.offline_projector_training_args.dataset.shuffle,
                                                lm_dir=student_dir,
                                                total_tokens=self.projection.offline_projector_training_args.total_tokens,
                                                force_cache_all=self.projection.offline_projector_training_args.force_cache_all,
                                                cache_outputs=False,
                                                num_workers=self.train_params.num_data_workers,
                                                device=self.train_params.device,
                                                torch_dtype=self.torch_dtype,)

        # create projector
        projector = OfflineProjector.create(
            teacher_cache_dir=teacher_cache_dir,
            student_cache_dir=student_cache_dir,
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            alignment_beta=self.projection.offline_projector_alignment_beta,
            max_epochs=self.projection.offline_projector_training_args.num_epochs,
            wd_min=self.hyperparams.weight_decay.mini,
            wd_max=self.hyperparams.weight_decay.maxi,
            mgn_min=self.hyperparams.max_grad_norm.mini,
            mgn_max=self.hyperparams.max_grad_norm.maxi,
            batch_sizes=self.hyperparams.batch_sizes,
            n_trials=self.projection.offline_projector_training_args.n_trials,
            seed=self.train_params.seed,
            reproducible_backend=self.train_params.reproducible_backend,
            num_gpus_per_trial=self.projection.offline_projector_training_args.num_gpus_per_trial,
            num_cpus_per_trial=self.projection.offline_projector_training_args.num_cpus_per_trial,
            torch_dtype=self.torch_dtype,
        )
        projector.save_pretrained(path=self.dirs.off_proj_model)
        # load_test = OfflineProjector.load_pretrained(save_path=self.dirs.off_proj_model)
        return projector.extract_single_direction(direction=self.projection.direction)

    def _init_online_projector(self):
        return OnlineProjector(teacher_dim=self.teacher.d_model,
                            student_dim=self.student.d_model,
                            direction=self.projection.direction,
                            torch_dtype=self.torch_dtype)
    def _init_randomized_projector(self):
        return RandomizedProjector(teacher_dim=self.teacher.d_model,
                            student_dim=self.student.d_model,
                            direction=self.projection.direction,
                            torch_dtype=self.torch_dtype)

    def init_projector(self) -> None:
        """Initialize projector for an experiment. kwargs are for init_offline_projector."""
        if self.projection:
            if self.projection.training == ProjectorTraining.ONLINE:
                p = self._init_online_projector()
            elif self.projection.training == ProjectorTraining.OFFLINE:
                p = self._init_offline_projector()
            elif self.projection.training == ProjectorTraining.RANDOMIZED:
                p = self._init_randomized_projector()
            else:
                raise ValueError(f"Invalid projection training specified: {self.projection.training}. Must be either 'online' or 'offline'.")
            self.projection.model = p
        return

    def init_teacher_by_task(self, task: TaskSpec) -> None:
        save_dir = f"{self.dirs.teacher_model}/{task.task_name}"
        teacher_weights_file = Path(gen_teacher_weights_file(task.task_name))
        if teacher_weights_file.exists():
            print(f"Loading pretrained teacher for task {task.task_name} at: {str(teacher_weights_file)}")
            self.teacher.pl = Teacher.load_pretrained(save_dir)

        else:
            t = Teacher.create(
                lm_name=self.teacher.model_name,
                task=task,
                lm_params=self.lm_params,
                hyperparams=self.hyperparams,
                training_params=self.train_params,
                torch_dtype=self.torch_dtype,
            )
            t.save_pretrained(path=save_dir)
            self.teacher.pl = t


    def init(self) -> None:
        """Initialize teachers, student, and projector"""
        self.init_projector()
        for task in self.tasks:
            self.init_teacher_by_task(task)

