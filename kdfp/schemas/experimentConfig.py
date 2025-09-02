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
from kdfp.core.init import init_projector

class ExperimentConfig(ConfigBase):
    name: str
    reduction: Literal["shallow", "narrow"]
    teacher: LM
    student: LM
    dirs: ExpDirConfig
    training_params: TrainingParamsConfig
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
            'projection': self.projection.to_config_dict() if self.projection else "None",
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
    
    def init(self) -> None:
        """Initialize teacher, student, and projector"""
        init_projector(teacher=self.teacher,
                       student=self.student,
                       projection=self.projection,
                       cache_dir=self.dirs.cache,
                       save_dir=self.dirs.projector_model,
                       projection=self.projection,
                       off_proj_train_args=self.projection.offline_projector_training_args,
                       train_params=self.training_params,
                       hyperparams=self.hyperparams,
                       teacher_dir=None,
                       student_dir=None,)

