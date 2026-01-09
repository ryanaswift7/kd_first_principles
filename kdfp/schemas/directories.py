import os
from typing import Optional

from pydantic import model_validator

from kdfp.schemas.configBase import ConfigBase


class BaseDirConfig(ConfigBase):
    root: str
    cache: str
    models: str
    scores: str
    ray: str
    lightning: str


class ExpDirConfig(ConfigBase):
    cache: Optional[str] = None
    off_proj_models: Optional[str] = None
    teacher_model: str
    teacher_scores: str
    student_model: str
    student_scores: str
    ray: str
    lightning: str

    @model_validator(mode="after")
    def make_paths_absolute(self):
        self.teacher_model = os.path.abspath(self.teacher_model)
        self.teacher_scores = os.path.abspath(self.teacher_scores)
        self.student_model = os.path.abspath(self.student_model)
        self.student_scores = os.path.abspath(self.student_scores)
        self.ray = os.path.abspath(self.ray)
        self.lightning = os.path.abspath(self.lightning)
        if self.cache:
            self.cache = os.path.abspath(self.cache)
        if self.off_proj_models:
            self.off_proj_models = os.path.abspath(self.off_proj_models)
        return self
