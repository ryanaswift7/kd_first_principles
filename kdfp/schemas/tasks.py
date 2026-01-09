from typing import Literal, Optional, Union

from pydantic import model_validator

from kdfp.schemas.configBase import ConfigBase

TASK_PROPERTIES = {
    "boolq": {"task_type": "binary", "test_available": False, "ninety_percent": 8487},
    "winogrande": {
        "task_type": "binary",
        "test_available": False,
        "ninety_percent": 36360,
    },
    "piqa": {"task_type": "binary", "test_available": False, "ninety_percent": 14490},
    "arc_challenge": {
        "task_type": "multiclass",
        "test_available": True,
        "ninety_percent": None,
    },
    "arc_easy": {
        "task_type": "multiclass",
        "test_available": True,
        "ninety_percent": None,
    },
    "hellaswag": {
        "task_type": "multiclass",
        "test_available": False,
        "ninety_percent": 35910,
    },
    "sciq": {"task_type": "multiclass", "test_available": True, "ninety_percent": None},
    # "logiqa2": {"task_type": "multiclass", "test_available": False, "ninety_percent": },
}


class TaskConfig(ConfigBase):
    dataset_name: str
    subset: Optional[str] = None


class TaskInstance(ConfigBase):
    name: str
    dataset_name: str
    subset: Optional[str] = None
    num_fewshot: int = 0
    batch_size: Union[int, str] = "auto"

    # These fields are now correctly populated by the model_validator
    task_type: Optional[Literal["binary", "multiclass"]] = None
    test_available: Optional[bool] = None
    ninety_percent: Optional[int] = None

    @model_validator(mode="after")
    def set_derived_fields_from_name(self) -> "TaskInstance":
        """
        Authoritatively sets task_type and test_available based on the task name.
        This runs after the model is initialized.
        """
        if self.name in TASK_PROPERTIES:
            properties = TASK_PROPERTIES[self.name]
            # Overwrite fields to ensure consistency based on the task name
            self.task_type = properties["task_type"]
            self.test_available = properties["test_available"]
            self.ninety_percent = properties["ninety_percent"]
        else:
            # If the name is not in our pre-configured map, raise an error.
            raise ValueError(
                f"Task '{self.name}' is not supported. Please add its properties "
                f"to the TASK_PROPERTIES dictionary."
            )
        return self
