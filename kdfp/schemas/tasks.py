from typing import Optional, Union, Dict, Any, Literal
from kdfp.schemas.configBase import ConfigBase

class TaskSpec(ConfigBase): 
    name: str
    # task_type: Literal["binary", "multiclass"]
    dataset: str
    subset: Optional[str] = None
    num_fewshot: int = 0
    batch_size: Union[int, str] = "auto"

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Return a standardized dictionary representation for configuration purposes.
        Excludes the heavy model objects (tl, hf).
        """
        config_dict = {
            'task_name': self.task_name,
            'num_fewshot': self.num_fewshot,
        }
        return config_dict