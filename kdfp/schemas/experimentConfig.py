from typing import Optional, Literal, List, Dict, Any, Union
from kdfp.schemas.objective_weights import ObjectiveWeights
from kdfp.schemas.tasks import TaskSpec
from kdfp.schemas.lm import LM
from kdfp.schemas.configBase import ConfigBase
from pathlib import Path
import json
from pydantic import model_validator
from kdfp.schemas.subConfigs import (ExpDirConfig, OffProjTrainArgsConfig, TrainingParamsConfig,
                                       HyperparamsConfig,
                                       ExpProjectionConfig)
from kdfp.core.projector import OfflineProjector, OnlineProjector, OFF_PROJ_STATE_DICT_FILE
from kdfp.utils.activation_caching import activation_caching
import torch.nn as nn

class ExperimentConfig(ConfigBase):
    name: str
    reduction: Literal["shallow", "narrow"]
    teacher: LM
    student: LM
    dirs: ExpDirConfig
    train_params: TrainingParamsConfig
    projection: Optional[ExpProjectionConfig]
    objective_weight_variant: str
    objective_weights: ObjectiveWeights
    jsd_beta: Optional[float]
    hyperparams: HyperparamsConfig
    tasks: List[TaskSpec]


    @model_validator(mode='after')
    def validate_and_transform_config(self) -> 'ExperimentConfig':
        if self.projection:
            if self.reduction == "shallow":
                # all of the fields are already none but I'd prefer
                # to also make the whole object None
                self.projection = None

            elif self.projection.direction == "online":
                self.projection.offline_projector_alignment_beta = None
        return self

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Return a complete configuration dictionary.
        Automatically uses each object's to_config_dict method.
        NOTE: does not include tasks, hyperparams, or training params
                bc they are all in the config file
        """
        config_dict = {
            'name': self.name,
            'dirs': self.dirs.to_config_dict(),
            'reduction': self.reduction,
            'teacher': self.teacher.to_config_dict(),
            'student': self.student.to_config_dict(),
            'projection_cfg': self.projection.to_config_dict() if self.projection else "None",
            'objective_weight_variant': self.objective_weight_variant,
            'objective_weights': self.objective_weights.to_config_dict(),
            'jsd_beta': self.jsd_beta,
            # 'tasks': [task.to_config_dict() for task in self.tasks]
        }
        return config_dict
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        Alternative method name for consistency.
        """
        return self.to_config_dict()
    
    def pretty_print_config(self) -> None:
        """Pretty print the configuration."""
        from pprint import pprint
        pprint(self.to_config_dict())

    def save_config_json(self, save_dir: Union[str, Path]) -> Path:
        """
        Save the configuration to a JSON file named 'config.json'.
        
        Args:
            save_dir: Directory where the config file should be saved
            
        Returns:
            Path to the saved JSON file
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / 'config.json'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_config_dict(), f, indent=4, ensure_ascii=False, default=str)
        
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
            print(f"Loading pretrained projector at: {str(weights_path)}.")
            p = OfflineProjector.load_pretrained(str(weights_path))
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
                                                num_workers=self.train_params.num_workers,
                                                device=self.train_params.device)
        
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
                                                num_workers=self.train_params.num_workers,
                                                device=self.train_params.device)

        # create projector
        projector = OfflineProjector.create(
            teacher_cache_dir=teacher_cache_dir,
            student_cache_dir=student_cache_dir,
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            alignment_beta=self.projection.offline_projector_alignment_beta,
            max_epochs=self.train_params.num_epochs,
            wd_min=self.hyperparams.weight_decay.mini,
            wd_max=self.hyperparams.weight_decay.maxi,
            mgn_min=self.hyperparams.max_grad_norm.mini,
            mgn_max=self.hyperparams.max_grad_norm.maxi,
            batch_sizes=self.hyperparams.batch_sizes,
            n_trials=self.projection.offline_projector_training_args.n_trials,
            save_dir=f"{self.dirs.off_proj_model}/projector_{self.projection.offline_projector_alignment_beta}",
            seed=self.train_params.seed,
            reproducible_backend=self.train_params.reproducible_backend,
            num_gpus_per_trial=self.projection.offline_projector_training_args.num_gpus_per_trial,
            num_cpus_per_trial=self.projection.offline_projector_training_args.num_cpus_per_trial,
        )
        # projector.save_pretrained(save_path=self.dirs.off_proj_model)
        # return projector.extract_single_direction(direction=self.projection.direction)

    def _init_online_projector(self):
        return OnlineProjector(teacher_dim=self.teacher.d_model,
                            student_dim=self.student.d_model,
                            direction=self.projection.direction)

    def init_projector(self) -> None:
        """Initialize projector for an experiment. kwargs are for init_offline_projector."""
        if self.projection:
            if self.projection.training == 'online':
                p = self._init_online_projector()
            elif self.projection.training == 'offline':
                p = self._init_offline_projector()
            else:
                raise ValueError(f"Invalid projection training specified: {self.projection.training}. Must be either 'online' or 'offline'.")
            self.projection.model = p
        return
    
    # def init_teachers(self) -> None:


    def init(self) -> None:
        """Initialize teachers, student, and projector"""
        self.init_projector()
        # self.init_teachers()

