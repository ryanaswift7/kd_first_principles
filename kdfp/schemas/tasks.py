from typing import Optional, Union, Dict, Any, Literal
from kdfp.schemas.configBase import ConfigBase
from pydantic import model_validator

class TaskSpec(ConfigBase): 
    task_name: str
    dataset_name: str
    subset: Optional[str] = None
    num_fewshot: int = 0
    batch_size: Union[int, str] = "auto"
    task_type: Optional[Literal["binary", "multiclass"]] = None
    
    @model_validator(mode='after')
    def set_task_type(self) -> 'TaskSpec':
        binary_tasks = ["boolq", "winogrande", "piqa"]
        multiple_choice_tasks = ["arc_challenge", "arc_easy", "hellaswag", "logiqa2", "sciq"]
        
        if self.task_name in binary_tasks:
            self.task_type = 'binary'
        elif self.task_name in multiple_choice_tasks:
            self.task_type = 'multiclass'
        else:
            raise ValueError(f"Task {self.task_name} is not supported.")
        return self